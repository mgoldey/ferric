"""Periodic bindings beyond `run_rhf_gamma` (src/pbc.rs): Gamma UHF / ROHF /
UKS / ROKS / RKS / MP2 / dRPA and k-point RHF / UHF / MP2 / dRPA.

Every numerical case reproduces a number already pinned in the Rust suite
(crates/ferric-pbc/tests/*.rs) on the same system, so a pass means the Python
surface reaches the validated driver with the same inputs (units, basis,
exxdiv, denominators, mesh); the physics validation itself (PySCF 2.13 AFTDF
pins, prototype grids) lives in those Rust tests, cited per case.

Systems (Bohr, as in crates/ferric-pbc/tests/common/mod.rs), converted to the
binding's Angstrom with the reciprocal of ferric's own factor:
    H atom at (0.3, 0.2, 0.1), H2 at (0.3, 0.2, 0.1)/(0.3, 0.2, 1.5),
    cubic a = 4 Bohr, STO-3G with PySCF's digits (see test_pbc_gamma.py),
    nuclear-attraction Ewald split 0.8 Bohr^-1 (the Rust suites'; the energy
    is split-independent to ~1e-9 anyway).

All H/H2 cases are nao <= 2 dense-AFT runs; the periodic-ECP (HI) cases are
nao 5 (k-mesh) / 15 (1x1x3 supercell) dense-AFT runs.
"""

from __future__ import annotations

import json

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR
OMEGA = 0.8 / BOHR_IN_ANGSTROM  # 0.8 Bohr^-1 in 1/Angstrom
A_BOHR = 4.0
H_ATOM = [(0.3, 0.2, 0.1)]
H2_ATOMS = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.5)]

# ---- pins (Hartree per cell), each copied from the named Rust test ----
# pbc_uhf.rs H_ATOM_E (PySCF pbc.scf.UHF, AFTDF): [none, ewald], <S2> 0.75.
H_UHF = {"none": -0.402177788224, "ewald": -0.756839973159}
# pbc_uks.rs H_SSF_E[0]: LDA, SSF 75x302, D = 10 Bohr, ewald; 19116 points.
H_UKS_LDA = -0.667583328116
H_SSF_NPTS = 19116
# pbc_rks.rs H2_SSF_75X302 LDA, ewald; 36234 points.
H2_RKS_LDA = -1.521871150768
H2_SSF_NPTS = 36234
# pbc_mp2.rs (PySCF pbc.mp.RMP2): unshifted, shifted.
H2_MP2 = {"unshifted": -5.891222456423e-3, "shifted": -3.881428851328e-3}
# pbc_drpa.rs (PySCF AFTDF dRPA): unshifted, shifted.
H2_DRPA = {"unshifted": -1.000052013959e-2, "shifted": -6.938130614440e-3}
# pbc_krhf.rs H2_112 (PySCF KRHF): Gamma-centred 1x1x2.
H2_KRHF_112 = {"none": -0.902683427348, "ewald": -1.354143879961}
# pbc_kuhf.rs H1X2 (PySCF KUHF): H atom doublet, 1x1x2, giant <S2> = 2.
H_KUHF_112 = {"none": -0.399399818915, "ewald": -0.625130045222}
# pbc_kcorr.rs KMP2_H2_112 (PySCF KMP2): (none, unshifted), (ewald, shifted).
H2_KMP2_112 = {"none": -2.8883196728367e-02, "ewald": -1.8228154905146e-02}

TOL = 1e-8

PYSCF_STO3G_H = {
    "name": "pyscf-sto-3g-H",
    "elements": {
        "1": {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["3.42525091", "0.62391373", "0.1688554"],
                    "coefficients": [["0.15432897", "0.53532814", "0.44463454"]],
                }
            ]
        }
    },
}


def _xyz(atoms_bohr):
    lines = [str(len(atoms_bohr)), "pbc bindings test (Angstrom)"]
    for x, y, z in atoms_bohr:
        lines.append(
            f"H {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
            f"{z * BOHR_IN_ANGSTROM!r}"
        )
    return "\n".join(lines) + "\n"


def _cubic(a_bohr=A_BOHR):
    a = a_bohr * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "pyscf-sto-3g-h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    return ferric.BasisSet.from_bse_json(str(p))


@pytest.fixture(scope="module")
def h_atom():
    return ferric.Molecule.from_xyz_string(_xyz(H_ATOM), multiplicity=2)


@pytest.fixture(scope="module")
def h2():
    return ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS))


# ── Gamma open shell ──


@pytest.mark.parametrize("fn", ["run_uhf_gamma", "run_rohf_gamma"])
@pytest.mark.parametrize("exx", ["none", "ewald"])
def test_h_atom_hf_matches_pinned_pyscf_uhf(h_atom, basis, fn, exx):
    # One electron: ROHF == UHF, so both reach pbc_uhf.rs's PySCF pin.
    r = getattr(ferric, fn)(h_atom, _cubic(), basis, exxdiv=exx, omega=OMEGA)
    assert r.converged and r.exxdiv == exx
    assert (r.nalpha, r.nbeta) == (1, 0)
    assert abs(r.energy - H_UHF[exx]) < TOL, f"{fn} {exx}: {r.energy!r}"
    assert abs(r.s2 - 0.75) < 1e-10
    assert r.ewald_start == ("staged" if exx == "ewald" else None)
    assert r.e_xc is None and r.functional is None


def test_uhf_ewald_minus_none_is_the_madelung_shift(h_atom, basis):
    lat = _cubic()
    e = {
        x: ferric.run_uhf_gamma(h_atom, lat, basis, exxdiv=x, omega=OMEGA)
        for x in ("none", "ewald")
    }
    # E_ewald - E_none = -v_M N_e / 2, N_e = 1.
    assert abs(e["ewald"].energy - e["none"].energy + e["ewald"].madelung / 2) < 1e-9
    assert e["ewald"].none_stage_energy is not None  # staged ran a none stage
    assert abs(e["ewald"].none_stage_energy - e["none"].energy) < 1e-9


@pytest.mark.parametrize("fn", ["run_uks_gamma", "run_roks_gamma"])
def test_h_atom_lda_matches_pinned_prototype_ssf_grid(h_atom, basis, fn):
    # pbc_uks.rs H_SSF_E (LDA); one electron: ROKS == UKS.
    r = getattr(ferric, fn)(h_atom, _cubic(), basis, "LDA", omega=OMEGA)
    assert r.converged and r.functional == "LDA"
    assert r.n_grid_points == H_SSF_NPTS
    assert abs(r.energy - H_UKS_LDA) < TOL, f"{fn}: {r.energy!r}"
    assert abs(r.electrons_on_grid - 1.0) < 1e-2
    assert r.exact_exchange_fraction == 0.0


def test_h2_rks_lda_matches_pinned_prototype_ssf_grid(h2, basis):
    r = ferric.run_rks_gamma(h2, _cubic(), basis, "LDA", omega=OMEGA)
    assert r.converged and r.n_grid_points == H2_SSF_NPTS
    assert abs(r.energy - H2_RKS_LDA) < TOL, f"{r.energy!r}"
    # Default cutoff: max(10 Bohr, covering bound) = 10 Bohr for this cube.
    assert abs(r.neighbour_cutoff - 10.0 * BOHR_IN_ANGSTROM) < 1e-9


# ── Gamma correlation ──

# (reference exxdiv, denominators, pin key): every combination is valid and
# the shift makes the reference exxdiv irrelevant to the correlation energy.
CORR_CASES = [
    ("none", "unshifted"),
    ("none", "shifted"),
    ("ewald", "shifted"),
    ("ewald", "unshifted"),
]


@pytest.mark.parametrize("exx,den", CORR_CASES)
def test_h2_mp2_matches_pinned_pyscf_rmp2(h2, basis, exx, den):
    r = ferric.run_mp2_gamma(h2, _cubic(), basis, exx, den, omega=OMEGA)
    assert r.method == "mp2" and r.denominators == den and r.exxdiv == exx
    assert abs(r.correlation_energy - H2_MP2[den]) < 1e-9, f"{r.correlation_energy!r}"
    assert abs(r.e_os + r.e_ss - r.correlation_energy) < 1e-14
    assert abs(r.energy - r.e_scf - r.correlation_energy) < 1e-14
    assert r.naux is None and (r.nocc_active, r.nvir) == (1, 1)


@pytest.mark.parametrize("exx,den", CORR_CASES)
def test_h2_drpa_matches_pinned_pyscf_drpa(h2, basis, exx, den):
    r = ferric.run_drpa_gamma(h2, _cubic(), basis, exx, den, omega=OMEGA)
    assert r.method == "drpa" and r.e_os is None
    assert r.quad_points is None  # dense = exact plasmon formula
    assert abs(r.correlation_energy - H2_DRPA[den]) < 1e-9, f"{r.correlation_energy!r}"


# ── k-point ──


@pytest.mark.parametrize("exx", ["none", "ewald"])
def test_h2_krhf_112_matches_pinned_pyscf_krhf(h2, basis, exx):
    r = ferric.run_rhf_kpts(
        h2,
        _cubic(),
        basis,
        (1, 1, 2),
        exxdiv=exx,
        omega=OMEGA,
        energy_conv=1e-13,
        grad_conv=1e-11,
    )
    assert r.converged and r.nk == 2 and r.mesh == (1, 1, 2)
    assert r.centring == "gamma" and len(r.mo_energy) == 2
    assert abs(r.energy - H2_KRHF_112[exx]) < TOL, f"{exx}: {r.energy!r}"
    assert r.lindep_total_kept == 4 and not r.lindep_near_noise_floor


@pytest.mark.parametrize("exx", ["none", "ewald"])
def test_h_atom_kuhf_112_matches_pinned_pyscf_kuhf(h_atom, basis, exx):
    r = ferric.run_uhf_kpts(
        h_atom,
        _cubic(),
        basis,
        (1, 1, 2),
        exxdiv=exx,
        omega=OMEGA,
        energy_conv=1e-13,
        grad_conv=1e-10,
    )
    assert r.converged and r.method == "uhf"
    assert abs(r.energy - H_KUHF_112[exx]) < TOL, f"{exx}: {r.energy!r}"
    assert abs(r.s2 - 2.0) < 1e-8  # giant determinant: two doublets, S = 1
    assert (r.nalpha, r.nbeta) == (1, 0)


@pytest.mark.parametrize("exx,den", [("none", "unshifted"), ("ewald", "shifted")])
def test_h2_kmp2_112_matches_pinned_pyscf_kmp2(h2, basis, exx, den):
    r = ferric.run_mp2_kpts(
        h2,
        _cubic(),
        basis,
        (1, 1, 2),
        exx,
        den,
        omega=OMEGA,
        energy_conv=1e-13,
        grad_conv=1e-11,
    )
    assert abs(r.correlation_energy - H2_KMP2_112[exx]) < 1e-9, (
        f"{exx}/{den}: {r.correlation_energy!r}"
    )
    assert abs(r.e_os + r.e_ss - r.correlation_energy) < 1e-14


@pytest.mark.parametrize("energy", ["plasmon", "quadrature"])
@pytest.mark.parametrize("den", ["unshifted", "shifted"])
def test_one_point_mesh_kdrpa_is_the_gamma_drpa_pin(h2, basis, energy, den):
    # pbc_kcorr.rs one_point_mesh_is_the_gamma_mp2_and_drpa: a 1x1x1 mesh is
    # the Gamma dRPA (plasmon <= 1e-12, 40-point quadrature <= 1e-11 on the
    # same orbitals); here each side converges its own SCF.
    r = ferric.run_drpa_kpts(
        h2,
        _cubic(),
        basis,
        (1, 1, 1),
        "ewald",
        den,
        omega=OMEGA,
        energy_conv=1e-13,
        grad_conv=1e-11,
        energy=energy,
    )
    assert r.drpa_energy == energy and len(r.per_q) == 1
    assert r.quad_points == (40 if energy == "quadrature" else None)
    assert abs(r.correlation_energy - H2_DRPA[den]) < TOL, f"{r.correlation_energy!r}"


# ── periodic ECP (crates/ferric-pbc/tests/pbc_ecp.rs) ──
#
# HI in a diag(6, 6, 7) Bohr cell, H (0.3, 0.2, 0.4), I (0.3, 0.2, 3.44):
# H STO-3G (PySCF digits) + the COMPACT I basis (one s 0.4653 + one Cartesian
# p 0.318) + the LANL2DZ ECP on I (46 core electrons, Z_eff = 7), i.e.
# pbc_ecp.rs `hi_basis(false)` transcribed to BSE JSON (same r^{n-2} term
# convention, same unit-self-overlap renormalisation). 8 valence electrons
# per cell, nao = 5.
#
# The compact case carries NO absolute pin in the Rust suite (the only pin is
# the ignored, ~minutes-long full-LANL2DZ 1x1x2 run), so this reproduces the
# Rust anchor `kmesh_rhf_with_ecp_equals_supercell` instead: k-mesh 1x1x3
# E/cell == Gamma E(1x1x3 supercell)/3 (Rust: <= 1e-11 at 1e-6 AFT precision;
# both bindings here use the same 1e-14 default). The Rust test shows the
# anchor catches a molecular-only V_ECP (> 1e-3). It is blind to a bare Z
# applied to both sides; that case cannot reach the SCF from Python because
# ferric-pbc's check_ecp_applied guard refuses it.

HI_A_BOHR = (6.0, 6.0, 7.0)
HI_ATOMS = [("H", (0.3, 0.2, 0.4)), ("I", (0.3, 0.2, 3.44))]

# LANL2DZ ECP for I (pbc_ecp.rs `lanl2dz_i_ecp`): (l, [(n, zeta, d), ...]).
# fmt: off
LANL2DZ_I_ECP = [
    (3, [(0, "1.0715702", "-0.0747621"), (1, "44.1936028", "-30.0811224"),
         (2, "12.9367609", "-75.3722721"), (2, "3.1956412", "-22.0563758"),
         (2, "0.8589806", "-1.6979585")]),
    (0, [(0, "127.9202670", "2.9380036"), (1, "78.6211465", "41.2471267"),
         (2, "36.5146237", "287.8680095"), (2, "9.9065681", "114.3758506"),
         (2, "1.9420086", "37.6547714")]),
    (1, [(0, "13.0035304", "2.2222630"), (1, "76.0331404", "39.4090831"),
         (2, "24.1961684", "177.4075002"), (2, "6.4053433", "77.9889462"),
         (2, "1.5851786", "25.7547641")]),
    (2, [(0, "40.4278108", "7.0524360"), (1, "28.9084375", "33.3041635"),
         (2, "15.6268936", "186.9453875"), (2, "4.1442856", "71.9688361"),
         (2, "0.9377235", "9.3630657")]),
]
# fmt: on


def _hi_compact_bse():
    def gto(l, exps, coefs):
        return {
            "function_type": "gto",
            "angular_momentum": [l],
            "exponents": exps,
            "coefficients": [coefs],
        }

    return {
        "name": "HI-compact",
        "elements": {
            "1": PYSCF_STO3G_H["elements"]["1"],
            "53": {
                "electron_shells": [
                    gto(0, ["0.4653"], ["1.0"]),
                    gto(1, ["0.318"], ["1.0"]),
                ],
                "ecp_electrons": 46,
                "ecp_potentials": [
                    {
                        "ecp_type": "scalar_ecp",
                        "angular_momentum": [l],
                        "r_exponents": [t[0] for t in terms],
                        "gaussian_exponents": [t[1] for t in terms],
                        "coefficients": [[t[2] for t in terms]],
                    }
                    for l, terms in LANL2DZ_I_ECP
                ],
            },
        },
    }


def _hi_xyz(n_cells=1):
    """HI repeated n_cells times along c (the 1 x 1 x n supercell), Angstrom."""
    lines = [str(2 * n_cells), "periodic HI (Angstrom)"]
    for m in range(n_cells):
        for sym, (x, y, z) in HI_ATOMS:
            zz = z + m * HI_A_BOHR[2]
            lines.append(
                f"{sym} {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
                f"{zz * BOHR_IN_ANGSTROM!r}"
            )
    return "\n".join(lines) + "\n"


def _hi_lattice(n_c=1):
    a, b, c = (x * BOHR_IN_ANGSTROM for x in HI_A_BOHR)
    return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, n_c * c]]


@pytest.fixture(scope="module")
def hi_basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "hi-compact-lanl2dz-ecp.json"
    p.write_text(json.dumps(_hi_compact_bse()))
    return ferric.BasisSet.from_bse_json(str(p))


@pytest.mark.parametrize("exx", ["none", "ewald"])
def test_hi_ecp_kmesh_rhf_equals_supercell_gamma_rhf(hi_basis, exx):
    cell = ferric.Molecule.from_xyz_string(_hi_xyz(1))
    sc = ferric.Molecule.from_xyz_string(_hi_xyz(3))
    rk = ferric.run_rhf_kpts(
        cell,
        _hi_lattice(1),
        hi_basis,
        (1, 1, 3),
        exxdiv=exx,
        omega=OMEGA,
        energy_conv=1e-13,
        grad_conv=1e-10,
    )
    rg = ferric.run_rhf_gamma(
        sc, _hi_lattice(3), hi_basis, exxdiv=exx, omega=OMEGA, density_conv=1e-10
    )
    assert rk.converged and rg.converged
    assert rk.nao == 5 and len(rk.mo_energy) == 3
    de = rk.energy - rg.energy / 3.0
    assert abs(de) <= 1e-9, f"{exx}: E_k/cell {rk.energy!r} E_sc/3 {rg.energy / 3.0!r}"


def test_charged_ecp_cell_is_still_a_hard_error(hi_basis):
    ion = ferric.Molecule.from_xyz_string(_hi_xyz(1), charge=2)
    with pytest.raises(ValueError, match="charged cell"):
        ferric.run_rhf_gamma(ion, _hi_lattice(1), hi_basis)
    with pytest.raises(ValueError, match="charged cell"):
        ferric.run_rhf_kpts(ion, _hi_lattice(1), hi_basis, (1, 1, 2))
    with pytest.raises(ValueError, match="charged cell"):
        ferric.run_uhf_gamma(
            ferric.Molecule.from_xyz_string(_hi_xyz(1), charge=1, multiplicity=2),
            _hi_lattice(1),
            hi_basis,
        )


# ── strict parsing and refusals ──


def test_open_shell_charged_cell_is_a_hard_error(basis):
    ion = ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS), charge=1, multiplicity=2)
    with pytest.raises(ValueError, match="charged cell"):
        ferric.run_uhf_gamma(ion, _cubic(), basis)
    with pytest.raises(ValueError, match="charged cell"):
        ferric.run_uhf_kpts(ion, _cubic(), basis, (1, 1, 2))


def test_closed_shell_drivers_refuse_an_open_shell(h_atom, basis):
    with pytest.raises(ValueError, match="multiplicity 2"):
        ferric.run_rks_gamma(h_atom, _cubic(), basis, "LDA")
    with pytest.raises(ValueError, match="multiplicity 2"):
        ferric.run_mp2_gamma(h_atom, _cubic(), basis, "ewald", "shifted")
    with pytest.raises(ValueError, match="multiplicity 2"):
        ferric.run_rhf_kpts(h_atom, _cubic(), basis, (1, 1, 2))


def test_ewald_start_is_strict_and_ewald_only(h_atom, basis):
    with pytest.raises(ValueError, match="ewald_start"):
        ferric.run_uhf_gamma(
            h_atom, _cubic(), basis, exxdiv="none", ewald_start="staged"
        )
    with pytest.raises(ValueError, match="ewald_start"):
        ferric.run_rohf_gamma(h_atom, _cubic(), basis, ewald_start="none-first")
    with pytest.raises(ValueError, match="ewald_start"):
        ferric.run_uhf_kpts(
            h_atom, _cubic(), basis, (1, 1, 2), exxdiv="none", ewald_start="direct"
        )


@pytest.mark.parametrize("bad", ["", "madelung", "shift", "Shifted "])
def test_bad_denominators_are_hard_errors(h2, basis, bad):
    with pytest.raises(ValueError, match="denominators"):
        ferric.run_mp2_gamma(h2, _cubic(), basis, "ewald", bad)
    with pytest.raises(ValueError, match="denominators"):
        ferric.run_drpa_kpts(h2, _cubic(), basis, (1, 1, 1), "ewald", bad)


def test_reference_exxdiv_and_denominators_are_required(h2, basis):
    with pytest.raises(TypeError):
        ferric.run_mp2_gamma(h2, _cubic(), basis)
    with pytest.raises(TypeError):
        ferric.run_mp2_kpts(h2, _cubic(), basis, (1, 1, 2), "ewald")


@pytest.mark.parametrize("bad", ["", "monkhorst-pack", "Gamma-centred", "mp2"])
def test_bad_centring_is_a_hard_error(h2, basis, bad):
    with pytest.raises(ValueError, match="centring"):
        ferric.run_rhf_kpts(h2, _cubic(), basis, (1, 1, 2), centring=bad)


def test_quad_points_is_refused_where_it_would_be_ignored(h2, basis):
    with pytest.raises(ValueError, match="quad_points"):
        ferric.run_drpa_gamma(h2, _cubic(), basis, "ewald", "shifted", quad_points=20)
    with pytest.raises(ValueError, match="quad_points"):
        ferric.run_drpa_kpts(
            h2,
            _cubic(),
            basis,
            (1, 1, 1),
            "ewald",
            "shifted",
            energy="plasmon",
            quad_points=20,
        )


@pytest.mark.parametrize("bad", ["", "rpa", "Plasmon2", "second_order"])
def test_bad_kdrpa_energy_is_a_hard_error(h2, basis, bad):
    with pytest.raises(ValueError, match="energy"):
        ferric.run_drpa_kpts(
            h2, _cubic(), basis, (1, 1, 1), "ewald", "shifted", energy=bad
        )


def test_rust_refusals_surface_as_value_errors(h_atom, h2, basis):
    # Unknown / refused functionals are refused by ferric-pbc itself.
    with pytest.raises(ValueError, match="run_uks_gamma"):
        ferric.run_uks_gamma(h_atom, _cubic(), basis, "not-a-functional")
    # A zero mesh dimension is refused by KPointMesh::new.
    with pytest.raises(ValueError, match="run_rhf_kpts"):
        ferric.run_rhf_kpts(h2, _cubic(), basis, (0, 1, 1))
    with pytest.raises(ValueError, match="neighbour_cutoff"):
        ferric.run_rks_gamma(h2, _cubic(), basis, "LDA", neighbour_cutoff=0.0)


def test_shared_jk_knob_policy_applies_to_every_driver(h2, h_atom, basis):
    with pytest.raises(ValueError, match="requires auxbasis"):
        ferric.run_rhf_kpts(h2, _cubic(), basis, (1, 1, 2), jk="rsgdf")
    with pytest.raises(ValueError, match="auxbasis is only used"):
        ferric.run_uhf_gamma(h_atom, _cubic(), basis, auxbasis="cc-pvdz-ri")
    with pytest.raises(ValueError, match="exxdiv"):
        ferric.run_mp2_kpts(h2, _cubic(), basis, (1, 1, 2), "vcut_sph", "shifted")
