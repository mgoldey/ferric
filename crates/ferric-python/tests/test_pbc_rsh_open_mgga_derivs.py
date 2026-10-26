"""Issue #498: the library routes wired to the Python bindings.

* ROKS with range-separated hybrids (energy): `run_roks_kpts` and
  `run_roks_gamma`.
* Closed-shell range-separated hybrid force and stress on the dense-AFT J/K:
  `run_rks_kpts` and `run_rks_gamma` `with_gradient` / `with_stress`.
* Open-shell UKS SCAN / r2SCAN force and stress (dense-AFT J/K):
  `run_uks_kpts` `with_gradient` / `with_stress`.
* Open-shell UKS range-separated hybrid FORCE (HSE06, CAM-B3LYP; dense-AFT
  J/K): `run_uks_kpts` and `run_uks_gamma` `with_gradient`.
* Closed-shell range-separated hybrid force and stress on the k-point RS-GDF
  J/K: `run_rks_kpts` `with_gradient` / `with_stress` with `jk="rsgdf"`.
* Open-shell range-separated hybrid derivatives (HSE06, CAM-B3LYP) on either
  J/K: the k-point UKS and ROKS force and stress (`run_uks_kpts`,
  `run_roks_kpts`), the Gamma UKS force and stress (`run_uks_gamma`) and the
  Gamma ROKS force (`run_roks_gamma`); and the Gamma RKS range-separated
  force and stress on RS-GDF (`run_rks_gamma`).
* What stays refused, by name: the Gamma ROKS range-separated stress (no
  library entry; the (1, 1, 1) k-point ROKS stress has it) and wB97X
  derivatives.

Anchors:

* Binding vs library: the numbers below are the CLI's `run_end` values for
  the same cell (bundled STO-3G, cubic a = 4 Bohr, omega 0.8 Bohr^-1, XC grid
  20x50 with a 14 Bohr cutoff, dense AFT, exxdiv ewald, 1x1x2 Gamma-centred
  mesh or the Gamma point), which crates/ferric-cli/tests/
  periodic_rsh_open_mgga_derivs.rs and periodic_kpoint_stress.rs prove equal
  to the library BIT FOR BIT. The geometries of the force pins are the
  optimizer-moved ones those runs logged (Bohr, exact repr). The binding
  builds its cell from Angstrom = Bohr * 0.529..., the CLI from the same
  value through its own path; the measured binding - library difference is
  recorded at each bar, which is ~10x it and far below the sibling-functional
  gap every pin also asserts (>= 5e-4).
* Independent constructions inside the binding: the k-point HSE06 (RKS and
  UKS) and UKS SCAN forces against a central FD of the binding's own
  energies; the Gamma CAM-B3LYP RKS force and stress and UKS force, the
  Gamma ROKS HSE06 force and the Gamma UKS HSE06 stress against the (1, 1, 1)
  mesh.
* The open-shell RSH pins (issue #498, second pass) are the CLI `run_end`
  values of crates/ferric-cli/tests/periodic_open_rsh_derivs.rs and
  validation_periodic_open_rsh_derivs.rs (zero optimizer steps: the input
  geometry, H3 at the parsed-xyz Bohr positions below), proven equal to the
  library BIT FOR BIT there; the Gamma UKS stress pins (the CLI has no Gamma
  stress) are the library values that file's
  `gamma_uks_rsh_stress_library_pins` prints.

Mutation ledger (binding source mutated, rebuilt, reverted; both KILLED):
`run_roks_gamma` with `exxdiv = none` forced misses the ROKS pin by 0.458 Ha;
the Gamma RSH stress returned transposed misses the 1x1x1 stress by 8.3e-5
(sigma is unsymmetrised on the atom grid).
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import (  # noqa: E402
    BOHR_IN_ANGSTROM,
    _fd,
    _lat,
    _mol,
)

CAM = "HYB_GGA_XC_CAM_B3LYP"
A = 4.0
OMEGA = 0.8 / BOHR_IN_ANGSTROM
GRID = dict(n_radial=20, n_angular=50, neighbour_cutoff=14.0 * BOHR_IN_ANGSTROM)
H2 = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5)]
H3 = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (0.3, 1.9, 0.4)]

# Library (= CLI run_end) values, see the module doc.
LIB_KROKS_HSE06 = -1.6217160984287937
LIB_KROKS_PBE0 = -1.6219656406405636
LIB_GROKS_CAM = -1.574336912992121
LIB_GROKS_B3LYP = -1.5495921505050014
# k-point RKS HSE06 (H2, 1x1x2) at the optimizer's geometry.
KRKS_X = [
    0.28962072444573533,
    0.21693700326959886,
    0.11795542079555987,
    0.3603792755542646,
    0.10306299672964517,
    1.4820445792044414,
]
KRKS_E = -1.351917468165642
KRKS_G = [
    0.006376126758084773,
    -0.010626126131219776,
    0.005229074775570414,
    -0.006376126758084767,
    0.010626126131219764,
    -0.005229074775570192,
]
# Gamma RKS CAM-B3LYP (H2).
GRKS_X = [
    0.29275250226011057,
    0.2116424251517566,
    -0.040728002935465676,
    0.3572474977398895,
    0.10835757484748752,
    1.640728002935466,
]
GRKS_E = -1.6257074299364547
GRKS_G = [
    0.0050572074946240156,
    -0.008100792347349433,
    0.06732024855095817,
    -0.005057207494623972,
    0.008100792347349353,
    -0.06732024855095776,
]
# k-point UKS SCAN (H3 doublet, 1x1x2).
KUKS_X = [
    0.30119913349778354,
    0.18172371570717294,
    0.02618184918062595,
    0.35245159921766334,
    0.09399054958084878,
    1.65551290307369,
    0.29634926728455346,
    1.9442857347112168,
    0.31830524774568336,
]
KUKS_E = -1.694375238002249
KUKS_G = [
    -0.0006087426581896983,
    0.024764433137510374,
    0.10510005279581994,
    -0.004595809180346683,
    0.03759756745262778,
    -0.21703784945310944,
    0.005204551838536309,
    -0.06236200059013376,
    0.11193779665729968,
]
# k-point UKS HSE06 (H3 doublet, 1x1x2, dense AFT).
KUKS_HSE_X = [
    0.3001949605595778,
    0.18512361201088973,
    0.028687735647317678,
    0.35336398924851115,
    0.09456740228633818,
    1.6564539534306768,
    0.29644105019191136,
    1.9403089857020166,
    0.3148583109220164,
]
KUKS_HSE_E = -1.6966079656966238
KUKS_HSE_G = [
    0.0006952682611394705,
    0.022267749304741898,
    0.10674890337085341,
    -0.006039927025646333,
    0.038022341001005905,
    -0.22575451347876005,
    0.005344658764506681,
    -0.06029009030574808,
    0.1190056101079084,
]
# Gamma UKS CAM-B3LYP (H3 doublet, dense AFT).
GUKS_X = [
    0.300146617455093,
    0.17979827867741846,
    0.051190547736766984,
    0.35783485352413785,
    0.08721507825828272,
    1.648267757129361,
    0.2920185290207688,
    1.9529866430635388,
    0.3005416951338663,
]
GUKS_E = -1.6468736874187138
GUKS_G = [
    0.00014251377901751167,
    0.03330087543737253,
    0.07388116688150345,
    -0.014485240451610686,
    0.056639085453457555,
    -0.22639007199330005,
    0.014342726672593405,
    -0.0899399608908317,
    0.1525089051117944,
]
# k-point RKS CAM-B3LYP on RS-GDF (H2, 1x1x2, cc-pVDZ-RI, default gdf_omega).
KRKS_GDF_X = [
    0.2899769884304344,
    0.2162485166912525,
    0.11819110532320959,
    0.36002301156956573,
    0.10375148330799154,
    1.4818088946767825,
]
KRKS_GDF_E = -1.3579350341963026
KRKS_GDF_G = [
    0.006322635213531635,
    -0.010389437042268518,
    0.004763743149493738,
    -0.006322635213531764,
    0.010389437042268591,
    -0.004763743149494487,
]
RSGDF = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")

# Issue #498, open-shell RSH derivatives: the CLI's parsed-xyz H3 / H2
# positions (Bohr), at which every pin below was evaluated (zero optimizer
# steps), and the CLI = library values.
X_H3 = [
    0.3,
    0.19999999999999998,
    0.09999999999999999,
    0.35,
    0.1199999999992441,
    1.4999999999999998,
    0.3,
    1.9,
    0.39999999999999997,
]
X_H2 = X_H3[:6]
# (tag, route, kpoints, rsgdf, functional, x, E, dE/dR)
OPEN_RSH_FORCES = [
    (
        "k ROKS HSE06",
        "roks",
        True,
        False,
        "HSE06",
        X_H3,
        -1.6217160984287924,
        [
            0.0011989843352957809,
            0.036480027026473516,
            0.21265141925510397,
            -0.006992040708774561,
            0.06141995533871902,
            -0.36880108267447925,
            0.005793056373478615,
            -0.09789998236519418,
            0.15614966341936926,
        ],
    ),
    (
        "k ROKS CAM-B3LYP RS-GDF",
        "roks",
        True,
        True,
        CAM,
        X_H3,
        -1.630327859693625,
        [
            0.002098814557951564,
            0.03423909842544136,
            0.2074955603174627,
            -0.008487917695619547,
            0.06032358574149925,
            -0.36726521650205224,
            0.006389103137667225,
            -0.09456268416694055,
            0.15976965618458278,
        ],
    ),
    (
        "k UKS CAM-B3LYP RS-GDF",
        "uks",
        True,
        True,
        CAM,
        X_H3,
        -1.6310067486339133,
        [
            0.002090003221944596,
            0.03446603537932972,
            0.20631937505224038,
            -0.008483676593427986,
            0.06031538629026999,
            -0.3679172739016368,
            0.006393673371483312,
            -0.09478142166958423,
            0.16159789884939157,
        ],
    ),
    (
        "Gamma ROKS CAM-B3LYP",
        "roks",
        False,
        False,
        CAM,
        X_H3,
        -1.574336912992121,
        [
            0.0007132677864258208,
            0.06561052276857068,
            0.12271573426884705,
            -0.013131280872817359,
            0.1011443041235046,
            -0.30006128506810587,
            0.012418013086390876,
            -0.16675482689207682,
            0.17734555079924977,
        ],
    ),
    (
        "Gamma ROKS HSE06 RS-GDF",
        "roks",
        False,
        True,
        "HSE06",
        X_H3,
        -1.5499685607074372,
        [
            3.851521263031465e-5,
            0.06770348004585922,
            0.12437046501951853,
            -0.011710407684267368,
            0.10285399610055798,
            -0.300184807501043,
            0.011671892471637357,
            -0.17055747614642064,
            0.1758143424815075,
        ],
    ),
    (
        "Gamma UKS CAM-B3LYP RS-GDF",
        "uks",
        False,
        True,
        CAM,
        X_H3,
        -1.5754918070632955,
        [
            0.0008397176360631434,
            0.06445757985289025,
            0.12387879945144904,
            -0.013282704940819784,
            0.10148428326030391,
            -0.30240189468214246,
            0.012442987304757604,
            -0.16594186311319878,
            0.1785230952306757,
        ],
    ),
    (
        "Gamma RKS HSE06 RS-GDF",
        "rks",
        False,
        True,
        "HSE06",
        X_H2,
        -1.5776585570742494,
        [
            0.007675787984522898,
            -0.012363388908638945,
            0.17190816072919118,
            -0.007675787984522946,
            0.012363388908638924,
            -0.1719081607291914,
        ],
    ),
]
# (tag, route, kpoints, rsgdf, functional, E, sigma row-major) at X_H3.
OPEN_RSH_STRESS = [
    (
        "k UKS HSE06",
        "uks",
        True,
        False,
        "HSE06",
        -1.6224626688847699,
        [
            0.010172508821464862,
            -5.979458705749496e-5,
            -7.551634573324573e-5,
            2.2265592867358865e-5,
            -0.0028142872768242566,
            0.0006654222733865025,
            -0.0001590379112009695,
            0.00045474016291087736,
            -0.010419367185719588,
        ],
    ),
    (
        "k ROKS HSE06",
        "roks",
        True,
        False,
        "HSE06",
        -1.6217160984287924,
        [
            0.010171734581412038,
            -5.9960864170078394e-5,
            -7.533854709877369e-5,
            2.2228989629024068e-5,
            -0.0027769588264296447,
            0.0006633847871327534,
            -0.00015884509753642554,
            0.0004524161708584239,
            -0.010426136374644493,
        ],
    ),
    (
        "k ROKS CAM-B3LYP RS-GDF",
        "roks",
        True,
        True,
        CAM,
        -1.630327859693625,
        [
            0.009374012611312202,
            -2.334295790686042e-5,
            -0.00011173048691405975,
            2.7664773736850203e-5,
            -0.003155836317187604,
            0.0006367122411249145,
            -0.00016536944767856464,
            0.0005086659183455922,
            -0.009963816416380873,
        ],
    ),
    (
        "k UKS CAM-B3LYP RS-GDF",
        "uks",
        True,
        True,
        CAM,
        -1.6310067486339133,
        [
            0.009375382574338254,
            -2.326717284673426e-5,
            -0.00011189154131630297,
            2.767028837332619e-5,
            -0.003192386984857386,
            0.0006381633560296673,
            -0.00016551656247670228,
            0.0005102603442384379,
            -0.00996014279051939,
        ],
    ),
    (
        "Gamma UKS HSE06",
        "uks",
        False,
        False,
        "HSE06",
        -1.5513072682853248,
        [
            0.009844533534466432,
            -5.4527968910358136e-5,
            -9.711154310969468e-5,
            3.849503929634633e-5,
            -0.018845108810043535,
            0.000872103794254751,
            -0.00017673785711825224,
            0.0006053574874198606,
            -0.0015261174092443376,
        ],
    ),
    (
        "Gamma UKS CAM-B3LYP RS-GDF",
        "uks",
        False,
        True,
        CAM,
        -1.5754918070632955,
        [
            0.009244335753996089,
            -1.1434950469432583e-5,
            -0.00013180352155890267,
            4.5930101422191205e-5,
            -0.018956309170521494,
            0.0008334749309097,
            -0.0001828865564665271,
            0.0006701955945700261,
            -0.0019885910287827926,
        ],
    ),
]
# CLI `[cell] stress = true` tensor (run_end, Hartree/Bohr^3) of the k-point
# RKS CAM-B3LYP on RS-GDF, H2 at (0.3, 0.2, 0.1), (0.3, 0.2, 1.6) Bohr (the
# periodic_kpoint_stress.rs cell, which that file proves equal to the library
# BIT FOR BIT), run on the xyz text `_mol` writes, so both sides parse the same
# Angstrom digits. That matters here: the RS-GDF range-separated stress is
# roundoff-limited (the attenuated fit keeps metric eigenvalues near the
# lindep cut, issue #512): the CLI's own 12-decimal xyz of the same Bohr
# positions (a ~1e-17 Angstrom difference) moves sigma_zz by 2.8e-9, and the
# BLAS kernel alone (OPENBLAS_CORETYPE Sandybridge / Prescott vs Haswell; the
# thread count gives 0) moves the RS-GDF force by up to 1.25e-8 and the stress
# by up to 1.65e-9. CI's OpenBLAS (0.3.20) is not the one that made these
# goldens (0.3.26), so the RS-GDF force and stress get their own bar,
# GDF_BAR = 1e-6: ~80x the worst measured kernel drift, ~570x below the B3LYP
# sibling gap (5.7e-4). Energies and the dense rows keep 1e-12.
KRKS_GDF_STRESS = [
    [0.005896909911451889, -2.5772410589560206e-18, 8.110865404359688e-19],
    [-3.0403972134496748e-18, 0.0058969099114519665, -4.9621959276285845e-17],
    [7.682996744004569e-19, -4.822552717660382e-17, -0.003282416530396363],
]
GDF_BAR = 1e-6
# Binding - library, measured: energy <= 7.1e-14 (ROKS; <= 8.9e-16 for the
# force runs), force <= 1.0e-15 (Gamma UKS CAM-B3LYP; RS-GDF CAM-B3LYP 0);
# RS-GDF stress 0. Bars ~10x the worst or more (and far below the >= 4.2e-4
# sibling gaps). FD: measured 2.8e-9 (RKS HSE06, h^2 floor of
# pbc_krsh_force.rs), 4.1e-10 (UKS HSE06), 7.6e-10 / 3.8e-10 (SCAN / r2SCAN);
# Gamma vs 1x1x1: RKS 1.1e-16 / 1.7e-17, UKS force 3.7e-12.
# Open-shell RSH pins (#498, 2026-10-26): energy <= 3.8e-15; dense force <=
# 2.3e-15, dense stress <= 7.5e-17; Gamma RS-GDF force <= 9.1e-15, stress
# 2.4e-15; k-point RS-GDF force 2.6e-9 / 3.8e-9 and stress 2.0e-10 / 8.0e-11
# (the roundoff-limited RS-GDF fit seeing the binding's repr-Angstrom vs the
# CLI's 12-decimal xyz, as for KRKS_GDF above; GDF_BAR). Gamma vs 1x1x1:
# ROKS HSE06 force 1.2e-14, UKS HSE06 stress 1.7e-13.
E_BAR = 1e-12
G_BAR = 1e-12
FD_H = 1e-4
FD_BAR = 2e-8
GAMMA_BAR = 1e-10


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


def _rows(flat):
    return [tuple(flat[3 * i : 3 * i + 3]) for i in range(len(flat) // 3)]


def _kw(**kw):
    return dict(omega=OMEGA, **GRID, **kw)


def test_roks_rsh_energies_equal_the_library(sto3g):
    k = ferric.run_roks_kpts(_mol(H3, 2), _lat(A), sto3g, "HSE06", (1, 1, 2), **_kw())
    g = ferric.run_roks_gamma(_mol(H3, 2), _lat(A), sto3g, CAM, **_kw())
    assert k.converged and g.converged
    assert k.method == "roks" and g.method == "roks"
    dk, dg = abs(k.energy - LIB_KROKS_HSE06), abs(g.energy - LIB_GROKS_CAM)
    print(f"ROKS binding - library: k HSE06 {dk:.2e}, Gamma CAM-B3LYP {dg:.2e}")
    assert dk < E_BAR and dg < E_BAR
    assert abs(LIB_KROKS_HSE06 - LIB_KROKS_PBE0) > 1e-4
    assert abs(LIB_GROKS_CAM - LIB_GROKS_B3LYP) > 1e-2
    # c_sr is reported as the fraction (HSE06: 0.25).
    assert k.exact_exchange_fraction == pytest.approx(0.25)


@pytest.mark.parametrize(
    "tag, run, functional, x, e, g, mult",
    [
        ("k RKS HSE06", "rks_k", "HSE06", KRKS_X, KRKS_E, KRKS_G, 1),
        ("Gamma RKS CAM-B3LYP", "rks_g", CAM, GRKS_X, GRKS_E, GRKS_G, 1),
        ("k UKS SCAN", "uks_k", "SCAN", KUKS_X, KUKS_E, KUKS_G, 2),
        ("k UKS HSE06", "uks_k", "HSE06", KUKS_HSE_X, KUKS_HSE_E, KUKS_HSE_G, 2),
        ("Gamma UKS CAM-B3LYP", "uks_g", CAM, GUKS_X, GUKS_E, GUKS_G, 2),
        (
            "k RKS CAM-B3LYP RS-GDF",
            "rks_k_gdf",
            CAM,
            KRKS_GDF_X,
            KRKS_GDF_E,
            KRKS_GDF_G,
            1,
        ),
    ],
)
def test_forces_equal_the_library(sto3g, tag, run, functional, x, e, g, mult):
    mol, lat = _mol(_rows(x), mult), _lat(A)
    if run == "rks_k":
        r = ferric.run_rks_kpts(
            mol, lat, sto3g, functional, (1, 1, 2), with_gradient=True, **_kw()
        )
    elif run == "rks_g":
        r = ferric.run_rks_gamma(
            mol, lat, sto3g, functional, with_gradient=True, **_kw()
        )
    elif run == "rks_k_gdf":
        r = ferric.run_rks_kpts(
            mol, lat, sto3g, functional, (1, 1, 2), with_gradient=True, **_kw(**RSGDF)
        )
    elif run == "uks_g":
        r = ferric.run_uks_gamma(
            mol, lat, sto3g, functional, with_gradient=True, **_kw()
        )
    else:
        r = ferric.run_uks_kpts(
            mol, lat, sto3g, functional, (1, 1, 2), with_gradient=True, **_kw()
        )
    de = abs(r.energy - e)
    dg = float(np.abs(r.gradient().ravel() - np.array(g)).max())
    print(f"{tag}: binding - library energy {de:.2e}, force {dg:.2e}")
    assert de < E_BAR and dg < (GDF_BAR if run == "rks_k_gdf" else G_BAR)


def test_kpoint_rsh_force_matches_fd_and_stress_runs(sto3g):
    def run(atoms, **kw):
        return ferric.run_rks_kpts(
            _mol(atoms), _lat(A), sto3g, "HSE06", (1, 1, 3), **_kw(**kw)
        )

    r = run(H2, with_gradient=True, with_stress=True)
    g = r.gradient()
    assert np.abs(g).max() > 1e-3
    fd = _fd(lambda at: run(at).energy, H2, ((0, 2), (1, 1)), FD_H)
    miss = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    print(f"k RKS HSE06 force vs FD: {miss:.2e}")
    assert miss < FD_BAR
    s = r.stress()
    assert s.shape == (3, 3) and np.abs(s).max() > 1e-6
    assert run(H2).energy == r.energy


def test_gamma_rsh_derivatives_equal_the_one_point_mesh(sto3g):
    kw = _kw(with_gradient=True, with_stress=True)
    g = ferric.run_rks_gamma(_mol(H2), _lat(A), sto3g, CAM, **kw)
    k = ferric.run_rks_kpts(_mol(H2), _lat(A), sto3g, CAM, (1, 1, 1), **kw)
    dg = float(np.abs(g.gradient() - k.gradient()).max())
    ds = float(np.abs(g.stress() - k.stress()).max())
    print(f"Gamma vs 1x1x1 CAM-B3LYP: force {dg:.2e}, stress {ds:.2e}")
    assert np.abs(g.gradient()).max() > 1e-3 and np.abs(g.stress()).max() > 1e-6
    assert dg < GAMMA_BAR and ds < GAMMA_BAR


def test_unvalidated_rsh_derivatives_are_refused_by_name(sto3g):
    # wB97X runs as an energy; its force / stress are not validated.
    msg = (
        r"HYB_GGA_XC_WB97X: the range-separated hybrid force and stress are "
        r"validated for HSE06 / CAM-B3LYP \(HYB_GGA_XC_CAM_B3LYP\) only"
    )
    wb = "HYB_GGA_XC_WB97X"
    for kw in ("with_gradient", "with_stress"):
        with pytest.raises(ValueError, match=rf"^run_rks_kpts: {kw}: .*{msg}"):
            ferric.run_rks_kpts(
                _mol(H2), _lat(A), sto3g, wb, (1, 1, 2), **_kw(**{kw: True})
            )
        with pytest.raises(
            ValueError, match=rf'^run_rks_gamma: functional "{wb}": .*{msg}'
        ):
            ferric.run_rks_gamma(_mol(H2), _lat(A), sto3g, wb, **_kw(**{kw: True}))


@pytest.mark.parametrize("functional", ["SCAN", "r2SCAN"])
def test_uks_meta_gga_force_matches_fd_and_stress_runs(sto3g, functional):
    def run(atoms, **kw):
        return ferric.run_uks_kpts(
            _mol(atoms, 2), _lat(A), sto3g, functional, (1, 1, 2), **_kw(**kw)
        )

    r = run(H3, with_gradient=True, with_stress=True)
    g = r.gradient()
    fd = _fd(lambda at: run(at).energy, H3, ((0, 2), (2, 1)), FD_H)
    miss = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    print(f"k UKS {functional} force vs FD: {miss:.2e}")
    assert np.abs(g).max() > 1e-3 and miss < FD_BAR
    assert r.stress().shape == (3, 3) and np.abs(r.stress()).max() > 1e-6


def test_uks_rsh_force_matches_fd(sto3g):
    def run(atoms, **kw):
        return ferric.run_uks_kpts(
            _mol(atoms, 2), _lat(A), sto3g, "HSE06", (1, 1, 2), **_kw(**kw)
        )

    r = run(H3, with_gradient=True)
    g = r.gradient()
    fd = _fd(lambda at: run(at).energy, H3, ((0, 2), (2, 1)), FD_H)
    miss = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    print(f"k UKS HSE06 force vs FD: {miss:.2e}")
    assert np.abs(g).max() > 1e-3 and miss < FD_BAR
    assert r.stress() is None


def test_gamma_uks_rsh_force_equals_the_one_point_mesh(sto3g):
    kw = _kw(with_gradient=True)
    g = ferric.run_uks_gamma(_mol(H3, 2), _lat(A), sto3g, CAM, **kw)
    k = ferric.run_uks_kpts(_mol(H3, 2), _lat(A), sto3g, CAM, (1, 1, 1), **kw)
    de, dg = abs(g.energy - k.energy), float(np.abs(g.gradient() - k.gradient()).max())
    print(f"Gamma vs 1x1x1 UKS CAM-B3LYP: energy {de:.2e}, force {dg:.2e}")
    assert np.abs(g.gradient()).max() > 1e-3
    assert de < GAMMA_BAR and dg < GAMMA_BAR


def test_rsgdf_rks_rsh_stress_equals_the_cli(sto3g):
    # periodic_kpoint_stress.rs `rks_cam_b3lyp_rsgdf_equals_library` cell.
    h2 = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.6)]
    r = ferric.run_rks_kpts(
        _mol(h2), _lat(A), sto3g, CAM, (1, 1, 2), with_stress=True, **_kw(**RSGDF)
    )
    d = float(np.abs(r.stress() - np.array(KRKS_GDF_STRESS)).max())
    print(f"k RKS CAM-B3LYP RS-GDF stress binding - CLI: {d:.2e}")
    assert np.abs(r.stress()).max() > 1e-6 and d < GDF_BAR


def _open_run(route, kpoints, mol, functional, **kw):
    fn = {
        ("rks", True): ferric.run_rks_kpts,
        ("uks", True): ferric.run_uks_kpts,
        ("roks", True): ferric.run_roks_kpts,
        ("rks", False): ferric.run_rks_gamma,
        ("uks", False): ferric.run_uks_gamma,
        ("roks", False): ferric.run_roks_gamma,
    }[(route, kpoints)]
    mesh = ((1, 1, 2),) if kpoints else ()
    return fn(mol, _lat(A), ferric.BasisSet.bundled("sto-3g"), functional, *mesh, **kw)


@pytest.mark.parametrize(
    "tag, route, kpoints, gdf, functional, x, e, g",
    OPEN_RSH_FORCES,
    ids=[r[0] for r in OPEN_RSH_FORCES],
)
def test_open_shell_rsh_forces_equal_the_library(
    tag, route, kpoints, gdf, functional, x, e, g
):
    mol = _mol(_rows(x), 1 if route == "rks" else 2)
    extra = RSGDF if gdf else {}
    r = _open_run(route, kpoints, mol, functional, **_kw(with_gradient=True, **extra))
    de = abs(r.energy - e)
    dg = float(np.abs(r.gradient().ravel() - np.array(g)).max())
    print(f"{tag}: binding - library energy {de:.2e}, force {dg:.2e}")
    assert de < E_BAR and dg < (GDF_BAR if gdf else G_BAR)


@pytest.mark.parametrize(
    "tag, route, kpoints, gdf, functional, e, sigma",
    OPEN_RSH_STRESS,
    ids=[r[0] for r in OPEN_RSH_STRESS],
)
def test_open_shell_rsh_stress_equals_the_library(
    tag, route, kpoints, gdf, functional, e, sigma
):
    extra = RSGDF if gdf else {}
    mol = _mol(_rows(X_H3), 2)
    r = _open_run(route, kpoints, mol, functional, **_kw(with_stress=True, **extra))
    de = abs(r.energy - e)
    ds = float(np.abs(r.stress().ravel() - np.array(sigma)).max())
    print(f"{tag}: binding - library energy {de:.2e}, stress {ds:.2e}")
    assert np.abs(r.stress()).max() > 1e-6
    assert de < E_BAR and ds < (GDF_BAR if gdf else G_BAR)


def test_gamma_open_shell_rsh_derivatives_equal_the_one_point_mesh(sto3g):
    # Gamma ROKS force and Gamma UKS stress vs the binding's own (1, 1, 1)
    # k-point ROKS / UKS derivative (dense AFT): an independent construction.
    mol, lat = _mol(H3, 2), _lat(A)
    g = ferric.run_roks_gamma(mol, lat, sto3g, "HSE06", **_kw(with_gradient=True))
    k = ferric.run_roks_kpts(
        mol, lat, sto3g, "HSE06", (1, 1, 1), **_kw(with_gradient=True)
    )
    dg = float(np.abs(g.gradient() - k.gradient()).max())
    gs = ferric.run_uks_gamma(mol, lat, sto3g, "HSE06", **_kw(with_stress=True))
    ks = ferric.run_uks_kpts(
        mol, lat, sto3g, "HSE06", (1, 1, 1), **_kw(with_stress=True)
    )
    ds = float(np.abs(gs.stress() - ks.stress()).max())
    print(f"Gamma vs 1x1x1: ROKS HSE06 force {dg:.2e}, UKS HSE06 stress {ds:.2e}")
    assert np.abs(g.gradient()).max() > 1e-3 and np.abs(gs.stress()).max() > 1e-6
    assert dg < GAMMA_BAR and ds < GAMMA_BAR


def test_open_shell_rsh_derivatives_that_stay_refused(sto3g):
    mol3, lat = _mol(H3, 2), _lat(A)
    # The Gamma ROKS range-separated stress has no library entry.
    with pytest.raises(
        ValueError,
        match=r'^run_roks_gamma: functional "HSE06": .*HSE06: the Gamma-point ROKS '
        r"range-separated hybrid stress has no library entry; the k-point ROKS "
        r"stress on a 1x1x1 mesh \(the Gamma point\) differentiates it, and the "
        r"Gamma ROKS force is available",
    ):
        ferric.run_roks_gamma(mol3, lat, sto3g, "HSE06", **_kw(with_stress=True))
    # wB97X: energy only, every open-shell derivative refused by name.
    wb = "HYB_GGA_XC_WB97X"
    msg = rf"{wb}: the range-separated hybrid force and stress are validated"
    for kw in ("with_gradient", "with_stress"):
        for fn in (ferric.run_uks_kpts, ferric.run_roks_kpts):
            with pytest.raises(ValueError, match=rf"^{fn.__name__}: {kw}: .*{msg}"):
                fn(mol3, lat, sto3g, wb, (1, 1, 2), **_kw(**{kw: True}))
        with pytest.raises(ValueError, match=rf"^run_uks_gamma: .*{msg}"):
            ferric.run_uks_gamma(mol3, lat, sto3g, wb, **_kw(**{kw: True}))
    with pytest.raises(ValueError, match=rf"^run_roks_gamma: .*{msg}"):
        ferric.run_roks_gamma(mol3, lat, sto3g, wb, **_kw(with_gradient=True))
