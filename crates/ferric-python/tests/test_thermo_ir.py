"""FrequencyResult.thermo against PySCF hessian.thermo, and
FrequencyResult.ir_intensities against PySCF dipoles.

Thermochemistry: PySCF's ``harmonic_analysis`` and ``thermo`` run on ferric's
own Cartesian Hessian, geometry and masses (ferric's isotope-averaged weights,
installed in place of PySCF's table), with PySCF's ``nist`` constants set to
the CODATA 2018 values ferric uses. Every E, H, T*S and G term must then
agree to 1e-8 Hartree. With PySCF's stock CODATA 2014 constants the
difference is reported, not asserted: k_B differs by 3.5e-7 relative, which
moves T*S_trans (~0.017 Ha at 298 K) by ~6e-9 Ha.

IR: PySCF RHF dipoles at displaced geometries (a third implementation of the
dipole route, independent of ferric's SCF), contracted with PySCF's own
normal modes from ferric's Hessian and PySCF's 974.88 km/mol per e^2/u
convention, against ferric's field route.
"""

import types
import warnings

import numpy as np
import pytest

import ferric

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402
from pyscf.data import nist  # noqa: E402
from pyscf.hessian import thermo as pthermo  # noqa: E402

MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
BOHR_ANG = 0.52917721092

CODATA_2018 = {
    "BOLTZMANN": 1.380649e-23,
    "PLANCK": 6.62607015e-34,
    "HBAR": 6.62607015e-34 / (2 * np.pi),
    "AVOGADRO": 6.02214076e23,
    "ATOMIC_MASS": 1.66053906660e-27,
    "HARTREE2J": 4.3597447222071e-18,
}


def _pyscf_mol(symbols, coords_ang, basis, spin=0):
    atom = [(s, tuple(c)) for s, c in zip(symbols, coords_ang)]
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", spin=spin, verbose=0)
    masses = np.array([MASS[s] for s in symbols])
    mol.atom_mass_list = lambda isotope_avg=False: masses
    return mol, masses


def _pyscf_thermo(mol, masses, hess_flat, e0, t, p):
    natm = mol.natm
    h4 = np.asarray(hess_flat).reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    ha = pthermo.harmonic_analysis(mol, h4, mass=masses)
    model = types.SimpleNamespace(mol=mol, e_tot=e0)
    return ha, pthermo.thermo(model, ha["freq_au"], t, p)


def _compare(ours, theirs, t):
    worst = 0.0
    for part, key in (
        ("electronic", "elec"),
        ("translational", "trans"),
        ("rotational", "rot"),
        ("vibrational", "vib"),
        ("total", "tot"),
    ):
        for q, pk, scale in (
            ("energy", "E", 1.0),
            ("enthalpy", "H", 1.0),
            ("entropy", "S", t),
            ("gibbs", "G", 1.0),
        ):
            mine = ours[part][q] * scale
            ref = theirs[f"{pk}_{key}"][0] * scale
            worst = max(worst, abs(mine - ref))
    worst = max(worst, abs(ours["zpe"] - theirs["ZPE"][0]))
    return worst


class _Codata2018:
    def __enter__(self):
        self.saved = {k: getattr(nist, k) for k in CODATA_2018}
        for k, v in CODATA_2018.items():
            setattr(nist, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(nist, k, v)


WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
CO2 = "3\n\nC 0 0 0\nO 0 0 1.16\nO 0 0 -1.16\n"
OH = "2\n\nO 0 0 0\nH 0 0 0.97\n"


@pytest.fixture(scope="module")
def water_run():
    mol = ferric.run_optimize(ferric.Molecule.from_xyz_string(WATER), "sto-3g").mol()
    return mol, ferric.run_frequencies(mol, "sto-3g", scf_density_conv=1e-10, scf_energy_conv=1e-10)


@pytest.mark.parametrize(
    ("xyz", "kw", "sigma", "spin", "points"),
    [
        (WATER, {}, 2, 0, [(298.15, 101325.0), (500.0, 2.0e5)]),
        (CO2, {}, 2, 0, [(298.15, 101325.0)]),
        (OH, {"multiplicity": 2, "reference": "uhf"}, 1, 1, [(298.15, 101325.0)]),
    ],
    ids=["water", "co2-linear", "oh-doublet"],
)
def test_thermo_matches_pyscf_on_the_same_hessian(xyz, kw, sigma, spin, points, water_run):
    if xyz is WATER:
        mol, fr = water_run
    else:
        mult = kw.get("multiplicity", 1)
        m0 = ferric.Molecule.from_xyz_string(xyz, multiplicity=mult)
        ref = kw.get("reference", "rhf")
        mol = ferric.run_optimize(m0, "sto-3g", reference=ref).mol()
        fr = ferric.run_frequencies(
            mol, "sto-3g", reference=ref, scf_density_conv=1e-10, scf_energy_conv=1e-10
        )
    pmol, masses = _pyscf_mol(mol.symbols(), mol.coords(), "sto-3g", spin=spin)
    assert pthermo.rotational_symmetry_number(pmol) == sigma
    for t, p in points:
        ours = fr.thermo(temperature=t, pressure=p, symmetry_number=sigma)
        with _Codata2018():
            ha, theirs = _pyscf_thermo(pmol, masses, fr.cartesian_hessian, fr.energy, t, p)
        wn = np.sort(ha["freq_wavenumber"].real)
        dnu = np.abs(wn - np.sort(fr.frequencies)).max()
        worst = _compare(ours, theirs, t)
        _, stock = _pyscf_thermo(pmol, masses, fr.cartesian_hessian, fr.energy, t, p)
        print(
            f"T={t} P={p}: max |ferric - PySCF| {worst:.2e} Ha (CODATA 2018), "
            f"{_compare(ours, stock, t):.2e} Ha (PySCF stock constants); "
            f"max |dnu| {dnu:.2e} cm-1"
        )
        assert worst < 1e-8, worst
        assert ours["rotor"] == ("linear" if fr.is_linear else "nonlinear")


def test_wrong_symmetry_number_is_visible(water_run):
    """sigma = 1 for C2v water: S_rot too large by R ln 2, G too low by RT ln 2."""
    _, fr = water_run
    one, two = fr.thermo(symmetry_number=1), fr.thermo(symmetry_number=2)
    r = 1.380649e-23 / 4.3597447222071e-18
    ds = one["rotational"]["entropy"] - two["rotational"]["entropy"]
    assert abs(ds - r * np.log(2)) < 1e-16
    assert abs((two["total"]["gibbs"] - one["total"]["gibbs"]) - 298.15 * r * np.log(2)) < 1e-12


def test_thermo_refuses_bad_inputs(water_run):
    _, fr = water_run
    for kw in ({"temperature": 0.0}, {"pressure": -1.0}, {"symmetry_number": 0}):
        with pytest.raises(ValueError):
            fr.thermo(**kw)


def _pyscf_apt(symbols, coords_ang, basis, d):
    """Dipole derivatives (e) from PySCF RHF dipoles at +-d Bohr."""
    base = np.array(coords_ang) / BOHR_ANG
    n = base.size
    apt = np.zeros((3, n))
    for i in range(n):
        mus = []
        for s in (1.0, -1.0):
            x = base.copy().ravel()
            x[i] += s * d
            m = gto.M(
                atom=[(sy, tuple(c)) for sy, c in zip(symbols, x.reshape(-1, 3))],
                basis=basis,
                unit="Bohr",
                verbose=0,
            )
            mf = scf.RHF(m)
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-9
            mf.kernel()
            assert mf.converged
            mus.append(mf.dip_moment(unit="AU", verbose=0))
        apt[:, i] = (mus[0] - mus[1]) / (2 * d)
    return apt


def test_ir_intensities_match_pyscf_dipoles(water_run):
    """Bar: the PySCF route's own Richardson truncation estimate |A(2d) - A(d)|/3
    (x2) plus the field route's measured disagreement with the dipole route
    (tests/ir_intensity_routes.rs, RHF water/6-31G: below 1e-5 e on P), both
    propagated to intensities through |dA| <= C (2|v| + b) b."""
    mol, fr = water_run
    ours = np.array(fr.ir_intensities())
    assert fr.ir_intensities() == list(ours)  # cached
    symbols, coords = mol.symbols(), mol.coords()
    pmol, masses = _pyscf_mol(symbols, coords, "sto-3g")
    with _Codata2018():
        ha, _ = _pyscf_thermo(pmol, masses, fr.cartesian_hessian, fr.energy, 298.15, 101325.0)
    # PySCF norm_mode: (mode, atom, xyz), = M^-1/2 u with M in u.
    modes = ha["norm_mode"].real.reshape(len(ha["freq_au"]), -1)
    order = np.argsort(ha["freq_wavenumber"].real)
    modes = modes[order]
    d = 5e-3
    p1, p2 = _pyscf_apt(symbols, coords, "sto-3g", d), _pyscf_apt(symbols, coords, "sto-3g", 2 * d)
    t = np.abs(p2 - p1).max() / 3.0
    b_p = 2.0 * t + 2e-5
    c_amu = 974.8801118
    theirs = []
    bounds = []
    for xk in modes:
        v = p1 @ xk
        theirs.append(c_amu * v @ v)
        b = b_p * np.sqrt(3) * np.abs(xk).sum()
        bounds.append(c_amu * (2 * np.linalg.norm(v) + b) * b)
    theirs = np.array(theirs)
    print(f"ferric {ours} vs PySCF dipoles {theirs} (bounds {bounds}, Richardson t {t:.1e})")
    assert len(ours) == len(theirs) == 3
    assert np.all(np.abs(ours - theirs) <= np.array(bounds))


def test_imaginary_modes_are_left_out_with_a_warning():
    """Planar NH3 (D3h) is a saddle: one imaginary umbrella mode."""
    nh3 = "4\n\nN 0 0 0\nH 1.0 0 0\nH -0.5 0.8660254 0\nH -0.5 -0.8660254 0\n"
    fr = ferric.run_frequencies(ferric.Molecule.from_xyz_string(nh3), "sto-3g")
    n_imag = sum(f < 0 for f in fr.frequencies)
    assert n_imag >= 1
    with pytest.warns(RuntimeWarning, match="imaginary"):
        th = fr.thermo(symmetry_number=6)
    assert th["n_imaginary_excluded"] == n_imag
    assert len(th["warnings"]) == 1
    assert len(fr.ir_intensities()) == len(fr.frequencies)
