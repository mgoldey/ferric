"""Issue #498, "fixtures beyond H/STO-3G": binding anchors of the wired
periodic derivative routes on cells whose basis has p and d shells, and on
cells with a heavy (Li) atom.

Every #498 binding anchor before this file used bundled STO-3G H cells (s
functions only), so no binding test touched an l > 0 shell in a force or
stress contraction. The cells here are those of
test_pbc_rsh_open_mgga_derivs.py (H2 closed shell, H3 doublet, cubic a = 4
Bohr, dense-AFT J/K, exxdiv ewald, omega 0.8 Bohr^-1, XC grid 20x50 with a 14
Bohr cutoff, 1x1x2 Gamma-centred mesh or the Gamma point) with
* H_SP: STO-3G H plus one p shell (zeta 1.0), or
* H_SPD: H_SP plus one pure d shell (zeta 1.0),
which keep the dense-AFT G sphere of the H/STO-3G cells (it is set by the
tightest exponent); and LiH / the LiH2 doublet with bundled STO-3G (Li 2p on a
Z = 3 nucleus; Li 1s sets a ~10x larger G sphere, so those are validation
tier).

Anchors:
* PHYSICS (fast tier, p shells), each against a central FD of the binding's
  OWN energy (h = 1e-4 Bohr / strain): the k-point RKS HSE06 force and full
  stress tensor, the k-point UKS SCAN force and stress (zz, xy), the k-point
  UKS HSE06 force, the Gamma RKS CAM-B3LYP force and stress (xx, xz), the
  Gamma UKS CAM-B3LYP force. A force is checked as the directional derivative
  along one generic 3N direction (every atom along every axis, by different
  amounts per atom, so all three Cartesian branches enter at the cost of two
  energies), against g . d; each axis' share of g . d must be non-trivial.
  NOTE: "fast tier" is NOT PR-gating. CI runs this file only in the nightly
  coverage job, which is continue-on-error. The PR-gating physics anchor of
  the same path is the library FD in
  crates/ferric-cli/tests/periodic_heavy_anchors.rs (Gamma RKS CAM-B3LYP force
  on the p-shell cell; it kills the mutant below).
* GLUE (validation tier, needs the CLI): the binding equals the CLI's
  `run_end` numbers on the same p-shell cell, and the CLI equals the library
  bit for bit (crates/ferric-cli/tests/periodic_heavy_anchors.rs), so the
  binding equals the library. No golden constants: both sides run here.
* d shells and Li, plus the k-point UKS r2SCAN stress on the p-shell cell
  (validation tier): the same FD anchors.
* BASIS GUARD: each custom-basis fixture runs a molecular RHF on H2 and
  checks the density is 8x8 (s+p) or 18x18 (s+p+d), so a reader that dropped
  the polarisation shells fails the fixture instead of running s-only cells.

The `validation` marker here means "too slow for the fast tier, or needs the
CLI binary" (the tier this repo runs with -m validation), not the marker's
pyproject wording "compares against a committed external reference": these
tests compare ferric with itself (FD of its own energy, binding vs CLI).

ARTIFACT HYPOTHESIS (written before measuring). The derivative of a Cartesian
Gaussian x^l exp(-a x^2) is 2a x^(l+1) - l x^(l-1) exp(...): an s shell sees
only the raising term, so a defect in the lowering term (a dropped or
mis-scaled `l F[l-1]`, an l-dependent normalisation of the derivative block,
a wrong cart-to-pure map in the derivative path) is INVISIBLE to every
H/STO-3G anchor and to every bit-equal binding-vs-library anchor (both sides
call the same library). If such a defect exists the FD of the energy (which
never differentiates a basis function) disagrees with the analytic force /
stress by roughly the p-population times the missing term, i.e. 1e-3..1e-2
Ha/Bohr here; if the p-shell path is right, the FD misses at the h^2 / grid
floor of the H cells (1e-9..1e-8). The two are 5+ decades apart, so the
experiment distinguishes them.

MEASURED (2026-10-26, release build, OPENBLAS_NUM_THREADS=1; the
OPENBLAS_CORETYPE=Prescott kernel moves every miss by < 2e-11). Forces are
|g . d - FD| along the generic direction:
* p shells: 3.4e-10 (k RKS HSE06 force), 2.0e-8 (k RKS HSE06 stress, six
  components), 7.4e-10 / 5.8e-9 (k UKS SCAN force / stress), 1.0e-8 (k UKS
  r2SCAN stress at h = 5e-5; 4.7e-8 at 1e-4, clean h^2), 4.5e-10 (k UKS
  HSE06 force), 5.2e-10 / 1.4e-8 (Gamma RKS CAM-B3LYP force / stress),
  5.3e-10 (Gamma UKS CAM-B3LYP force). Binding - CLI <= 1.6e-15 (forces: the
  CLI's moved geometry re-entered in Angstrom), stress 0.
* d shells: 3.2e-10 / 1.9e-8 (k RKS HSE06 force / stress), 2.8e-10 (k UKS
  SCAN force), 4.9e-10 / 1.3e-8 (Gamma RKS CAM-B3LYP force / stress).
* Li (STO-3G): 3.2e-11 / 5.3e-9 (LiH k RKS HSE06 force / stress), 4.9e-10
  (LiH2 k UKS SCAN force).
No FD miss on a p, d or Li cell exceeds the h^2 / grid floor of the
H/STO-3G cells: no defect found. The hypothesis' magnitude for a defect was
too high (the p shells here carry a small polarisation population), but
the mutants below still miss by 300x+ the worst floor.

RUNTIMES (release, one at a time): fast tier 10-34 s per test (~2.5 min
for the 7); validation 14-60 s per binding-vs-CLI test, 34 s r2SCAN stress,
52-125 s per d-shell test, 178 s LiH2, 609 s LiH: ~23 min in all (the
Python half of the validation tier runs on CI shard 4).

MUTATION LEDGER (library source mutated, extension rebuilt, reverted). The
mutant drops the p lowering term of the x-derivative (`if ax > 0` ->
`if ax > 1` in crates/ferric-pbc/src/pair_ft/residues.rs: s and d_xx-type
components keep their terms, every p_x component loses `-1 F[0]`):
* in the k-point pair-FT derivative (`residue_deriv_block`, the force; the
  Gamma range-separated force runs through it too), against the directional
  FD: k RKS HSE06 8.5e-6, k UKS SCAN 2.6e-4, k UKS HSE06 2.3e-5, Gamma RKS
  CAM-B3LYP 4.6e-5, Gamma UKS CAM-B3LYP 1.2e-4: all KILLED. The H/STO-3G FD
  anchors of test_pbc_rsh_open_mgga_derivs.py pass unchanged under the same
  mutant (2.8e-9 / 7.6e-10 / 4.1e-10): SURVIVED, which is the gap this file
  closes. The PR-gating Rust twin (periodic_heavy_anchors.rs) misses by
  4.6e-5: KILLED.
* in the k-point pair-FT strain derivative (`residue_strain_block`, the
  stress): k RKS HSE06 stress 1.3e-5, k UKS SCAN 1.1e-4, k UKS r2SCAN
  1.1e-4, Gamma RKS CAM-B3LYP 6.3e-6: KILLED. The previous Gamma-vs-(1, 1,
  1) stress check stayed at 5.9e-14 under this mutant: the Gamma
  range-separated derivatives run through the same k-point kernels, so that
  comparison is a routing check, not an independent construction; the
  Gamma anchors here are therefore FD anchors.
* With per-axis FD components, a component set without x (the first UKS
  choice, z and y only) let the force mutant survive; that is why forces
  are now checked along one direction with every axis' share of g . d
  asserted non-trivial. Stress component sets each include an x strain.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

import ferric  # noqa: E402  (conftest.py skips the suite without the extension)

ROOT = pathlib.Path(__file__).resolve().parents[3]
BOHR_IN_ANGSTROM = (
    1.0
    / (
        ferric.Molecule.from_xyz_string("1\n\nH 1.0 0.0 0.0\n", multiplicity=2)
    ).coords_bohr()[0][0]
)

CAM = "HYB_GGA_XC_CAM_B3LYP"
A = 4.0
OMEGA = 0.8 / BOHR_IN_ANGSTROM
GRID = dict(n_radial=20, n_angular=50, neighbour_cutoff=14.0 * BOHR_IN_ANGSTROM)
KMESH = (1, 1, 2)
H2 = (("H", "H"), ((0.3, 0.2, 0.1), (0.35, 0.12, 1.5)))
H3 = (("H", "H", "H"), ((0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (0.3, 1.9, 0.4)))
LIH = (("Li", "H"), ((0.3, 0.2, 0.1), (1.9, 1.4, 2.4)))
LIH2 = (("Li", "H", "H"), ((0.3, 0.2, 0.1), (1.9, 1.4, 2.4), (0.5, 2.9, 1.2)))

FD_H = 1e-4
# Bars (measured worst, see the module doc): force 7.4e-10 (p-shell UKS SCAN),
# stress 2.0e-8 (p-shell RKS HSE06, six components); binding - CLI 1.6e-15.
# The mutants miss by >= 6.3e-6.
FD_BAR = 1e-8
STRESS_BAR = 1e-7
CLI_BAR = 1e-12

# STO-3G H (the bundled digits) plus the polarisation shells, as the G94 the
# CLI anchors read (crates/ferric-cli/tests/heavy_anchors_cli/mod.rs).
_H_S = {
    "function_type": "gto",
    "angular_momentum": [0],
    "exponents": ["0.3425250914E+01", "0.6239137298E+00", "0.1688554040E+00"],
    "coefficients": [["0.1543289673E+00", "0.5353281423E+00", "0.4446345422E+00"]],
}


def _pol(ang):
    return {
        "function_type": "gto_spherical",
        "angular_momentum": [ang],
        "exponents": ["1.0000000000E+00"],
        "coefficients": [["1.0000000000E+00"]],
    }


def _basis(tmp, lmax):
    p = tmp / f"h_l{lmax}.json"
    shells = [_H_S] + [_pol(ang) for ang in range(1, lmax + 1)]
    p.write_text(json.dumps({"elements": {"1": {"electron_shells": shells}}}))
    bs = ferric.BasisSet.from_bse_json(str(p))
    # Basis guard: 1 + 3 (+ 5 pure d) functions per H, read back through a
    # molecular H2 RHF (the BasisSet object exposes no count).
    nbf = 2 * sum(2 * ang + 1 for ang in range(lmax + 1))
    shape = ferric.run_rhf(_mol(*H2, 1), bs).density().shape
    assert shape == (nbf, nbf), f"H l<={lmax}: density {shape}, want {nbf}"
    return bs


@pytest.fixture(scope="module")
def h_sp(tmp_path_factory):
    return _basis(tmp_path_factory.mktemp("basis"), 1)


@pytest.fixture(scope="module")
def h_spd(tmp_path_factory):
    return _basis(tmp_path_factory.mktemp("basis"), 2)


def _xyz(symbols, atoms_bohr):
    lines = [str(len(atoms_bohr)), "issue 498 heavy anchors (Angstrom)"]
    for s, (x, y, z) in zip(symbols, atoms_bohr):
        b = BOHR_IN_ANGSTROM
        lines.append(f"{s} {float(x * b)!r} {float(y * b)!r} {float(z * b)!r}")
    return "\n".join(lines) + "\n"


def _mol(symbols, atoms_bohr, mult):
    return ferric.Molecule.from_xyz_string(_xyz(symbols, atoms_bohr), multiplicity=mult)


def _lat(eps=None):
    f = np.eye(3) if eps is None else np.eye(3) + eps
    return ((A * np.eye(3)) @ f.T * BOHR_IN_ANGSTROM).tolist()


def _run(route, system, bs, functional, mult, atoms=None, eps=None, **kw):
    """One binding call; `eps` strains the lattice AND the atoms (r -> (1+eps) r)."""
    symbols, x0 = system
    atoms = [tuple(p) for p in (x0 if atoms is None else atoms)]
    if eps is not None:
        f = np.eye(3) + eps
        atoms = [tuple(f @ np.array(p)) for p in atoms]
    mol, lat = _mol(symbols, atoms, mult), _lat(eps)
    kw = dict(omega=OMEGA, **GRID, **kw)
    if route == "rks_k":
        return ferric.run_rks_kpts(mol, lat, bs, functional, KMESH, **kw)
    if route == "uks_k":
        return ferric.run_uks_kpts(mol, lat, bs, functional, KMESH, **kw)
    if route == "rks_1":
        return ferric.run_rks_kpts(mol, lat, bs, functional, (1, 1, 1), **kw)
    if route == "uks_1":
        return ferric.run_uks_kpts(mol, lat, bs, functional, (1, 1, 1), **kw)
    if route == "rks_g":
        return ferric.run_rks_gamma(mol, lat, bs, functional, **kw)
    if route == "uks_g":
        return ferric.run_uks_gamma(mol, lat, bs, functional, **kw)
    raise AssertionError(route)


# A generic displacement direction (rows: atoms), normalised over the whole
# 3N vector; the same rows as `generic_direction` in
# crates/ferric-cli/tests/heavy_anchors_cli/mod.rs. Per-atom components
# differ, so a net-force-preserving error in one axis cannot cancel.
_DIRECTION = np.array([[1.0, 0.7, 0.4], [-0.6, 0.9, -0.3], [0.5, -0.8, 0.2]])


def _force_vs_fd(route, system, bs, functional, mult):
    """|g . d - dE/dh| along the generic direction d, h = FD_H."""
    r = _run(route, system, bs, functional, mult, with_gradient=True)
    assert r.converged
    g = r.gradient()
    x0 = np.array(system[1], dtype=float)
    d = _DIRECTION[: len(x0)] / np.linalg.norm(_DIRECTION[: len(x0)])
    share = (g * d).sum(axis=0)
    assert (np.abs(share) > 1e-4).all(), f"an axis' share of g.d is trivial: {share}"
    e = [
        _run(route, system, bs, functional, mult, atoms=x0 + h * d).energy
        for h in (FD_H, -FD_H)
    ]
    return r, abs(float((g * d).sum()) - (e[0] - e[1]) / (2.0 * FD_H))


def _stress_vs_fd(route, system, bs, functional, mult, comps, h_fd=FD_H):
    """Omega (sigma_ab + sigma_ba) / 2 against the central FD of E under the
    symmetric strain eps_ab = eps_ba = h / 2 (eps_aa = h on the diagonal)."""
    r = _run(route, system, bs, functional, mult, with_stress=True)
    assert r.converged
    s = r.stress()
    vol = A**3
    worst = 0.0
    for a, b in comps:
        sym = vol * 0.5 * (s[a, b] + s[b, a])
        assert abs(sym) > 1e-5, f"trivial stress component {a}{b}: {sym!r}"
        e = []
        for h in (h_fd, -h_fd):
            eps = np.zeros((3, 3))
            eps[a, b] += h / 2.0
            eps[b, a] += h / 2.0
            e.append(_run(route, system, bs, functional, mult, eps=eps).energy)
        fd = (e[0] - e[1]) / (2.0 * h_fd)
        worst = max(worst, abs(sym - fd))
    return r, worst


ALL6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


# ───────────────────────────── p shells, fast tier (nightly in CI): physics ──


def test_kpoint_rks_hse06_force_p_shell_matches_fd(h_sp):
    _, miss = _force_vs_fd("rks_k", H2, h_sp, "HSE06", 1)
    print(f"k RKS HSE06 p-shell force vs FD: {miss:.2e}")
    assert miss < FD_BAR


def test_kpoint_rks_hse06_stress_p_shell_matches_fd(h_sp):
    _, miss = _stress_vs_fd("rks_k", H2, h_sp, "HSE06", 1, ALL6)
    print(f"k RKS HSE06 p-shell stress (6 components) vs FD: {miss:.2e}")
    assert miss < STRESS_BAR


def test_kpoint_uks_scan_force_p_shell_matches_fd(h_sp):
    _, miss = _force_vs_fd("uks_k", H3, h_sp, "SCAN", 2)
    print(f"k UKS SCAN p-shell force vs FD: {miss:.2e}")
    assert miss < FD_BAR


def test_kpoint_uks_scan_stress_p_shell_matches_fd(h_sp):
    _, miss = _stress_vs_fd("uks_k", H3, h_sp, "SCAN", 2, ((2, 2), (0, 1)))
    print(f"k UKS SCAN p-shell stress (zz, xy) vs FD: {miss:.2e}")
    assert miss < STRESS_BAR


def test_kpoint_uks_hse06_force_p_shell_matches_fd(h_sp):
    r, miss = _force_vs_fd("uks_k", H3, h_sp, "HSE06", 2)
    print(f"k UKS HSE06 p-shell force vs FD: {miss:.2e}")
    assert miss < FD_BAR


def test_gamma_rks_cam_b3lyp_p_shell_force_and_stress_match_fd(h_sp):
    # The Gamma range-separated derivatives run through the k-point kernels
    # on the one-point mesh, so "Gamma == (1, 1, 1)" is a routing check, not
    # an independent construction (the p-lowering mutants move both alike):
    # the physics anchor is the FD.
    _, fmiss = _force_vs_fd("rks_g", H2, h_sp, CAM, 1)
    _, smiss = _stress_vs_fd("rks_g", H2, h_sp, CAM, 1, ((0, 0), (0, 2)))
    print(
        f"Gamma RKS CAM-B3LYP p-shell: force vs FD {fmiss:.2e}, stress vs FD {smiss:.2e}"
    )
    assert fmiss < FD_BAR and smiss < STRESS_BAR


def test_gamma_uks_cam_b3lyp_p_shell_force_matches_fd(h_sp):
    _, miss = _force_vs_fd("uks_g", H3, h_sp, CAM, 2)
    print(f"Gamma UKS CAM-B3LYP p-shell force vs FD: {miss:.2e}")
    assert miss < FD_BAR


# ─────────────────────────────────── p shells, validation: binding vs CLI ──

_H_SP_G94 = """****
H     0
S   3   1.00
      0.3425250914E+01       0.1543289673E+00
      0.6239137298E+00       0.5353281423E+00
      0.1688554040E+00       0.4446345422E+00
P   1   1.00
      1.0000000000E+00       1.0000000000E+00
****
"""


def _cli_binary():
    """The CLI of this tree. `FERRIC_CLI` set but missing is a hard error (CI
    builds it and must not silently skip); unset, the local release build or
    `None`."""
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _cli_run_end(tmp, system, mult, kind, functional, kmesh, task):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    # Same tree on both sides, or the comparison is between two builds.
    ver = subprocess.run([str(cli), "--version"], capture_output=True, text=True)
    info = ferric.build_info()
    want = (f"commit: {info['commit']}", f"dirty: {str(info['dirty']).lower()}")
    if not all(w in ver.stdout for w in want):
        pytest.skip(
            f"ferric-cli ({cli}) and the extension are different builds: "
            f"{ver.stdout.strip()!r} vs {info}; rebuild both from this tree"
        )
    (tmp / "m.xyz").write_text(_xyz(*system))
    (tmp / "basis.g94").write_text(_H_SP_G94)
    extra = (
        "[optimize]\nmax_steps = 1\ng_max_thresh = 1e-9\ng_rms_thresh = 1e-9\n"
        "e_conv = 1e-14\n"
        if task == "optimize"
        else ""
    )
    toml = tmp / "in.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{tmp / "m.xyz"}"\nmultiplicity = {mult}\n'
        f'[basis]\npath = "{tmp / "basis.g94"}"\n'
        f'[method]\nkind = "{kind}"\ntask = "{task}"\n'
        f'[dft]\nfunctional = "{functional}"\n'
        f'[output]\njson = "{tmp / "run.jsonl"}"\n'
        f'[cell]\nunit = "bohr"\nlattice = [[{A}, 0.0, 0.0], [0.0, {A}, 0.0], '
        f"[0.0, 0.0, {A}]]\n"
        + (f"kmesh = {list(kmesh)}\n" if kmesh else "")
        + 'exxdiv = "ewald"\nomega = 0.8\nn_radial = 20\nn_angular = 50\n'
        "neighbour_cutoff = 14.0\n"
        + ("stress = true\n" if task == "energy" else "")
        + extra
    )
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    out = subprocess.run(
        [str(cli), str(toml)], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stdout + out.stderr
    for line in (tmp / "run.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("record") == "run_end":
            return rec
    raise AssertionError("no run_end record")


@pytest.mark.validation
@pytest.mark.parametrize(
    "tag, system, mult, kind, route, functional, kmesh",
    [
        ("k RKS HSE06", H2, 1, "rhf", "rks_k", "HSE06", KMESH),
        ("Gamma RKS CAM-B3LYP", H2, 1, "rhf", "rks_g", CAM, None),
        ("k UKS SCAN", H3, 2, "uhf", "uks_k", "SCAN", KMESH),
        ("k UKS HSE06", H3, 2, "uhf", "uks_k", "HSE06", KMESH),
        ("Gamma UKS CAM-B3LYP", H3, 2, "uhf", "uks_g", CAM, None),
    ],
)
def test_p_shell_force_equals_the_cli(
    tmp_path, h_sp, tag, system, mult, kind, route, functional, kmesh
):
    rec = _cli_run_end(tmp_path, system, mult, kind, functional, kmesh, "optimize")
    x = rec["extra"]["final_geometry_bohr"]
    at = [x[3 * i : 3 * i + 3] for i in range(len(x) // 3)]
    r = _run(route, system, h_sp, functional, mult, atoms=at, with_gradient=True)
    de = abs(r.energy - rec["energy"])
    g_cli = np.array(rec["extra"]["final_gradient_hartree_per_bohr"])
    dg = float(np.abs(r.gradient().ravel() - g_cli).max())
    print(f"{tag} p-shell: binding - CLI energy {de:.2e}, force {dg:.2e}")
    assert np.abs(g_cli).max() > 1e-3
    assert de < CLI_BAR and dg < CLI_BAR


@pytest.mark.validation
@pytest.mark.parametrize(
    "tag, system, mult, kind, route, functional",
    [
        ("k RKS HSE06", H2, 1, "rhf", "rks_k", "HSE06"),
        ("k UKS SCAN", H3, 2, "uhf", "uks_k", "SCAN"),
    ],
)
def test_p_shell_stress_equals_the_cli(
    tmp_path, h_sp, tag, system, mult, kind, route, functional
):
    rec = _cli_run_end(tmp_path, system, mult, kind, functional, KMESH, "energy")
    r = _run(route, system, h_sp, functional, mult, with_stress=True)
    s_cli = np.array(rec["extra"]["stress_hartree_per_bohr3"])
    de = abs(r.energy - rec["energy"])
    ds = float(np.abs(r.stress() - s_cli).max())
    print(f"{tag} p-shell: binding - CLI energy {de:.2e}, stress {ds:.2e}")
    assert np.abs(s_cli).max() > 1e-6
    assert de < CLI_BAR and ds < CLI_BAR


# ───────────────────────────────────────── d shells and Li: validation ──


@pytest.mark.validation
def test_kpoint_uks_r2scan_stress_p_shell_matches_fd(h_sp):
    # r2SCAN's zz strain curvature is larger: the miss is the FD truncation,
    # measured 1.9e-7 / 4.7e-8 / 1.0e-8 at h = 2e-4 / 1e-4 / 5e-5 (clean h^2),
    # so this one runs at h = 5e-5.
    _, miss = _stress_vs_fd(
        "uks_k", H3, h_sp, "r2SCAN", 2, ((2, 2), (0, 1)), h_fd=FD_H / 2
    )
    print(f"k UKS r2SCAN p-shell stress (zz, xy) vs FD: {miss:.2e}")
    assert miss < STRESS_BAR


@pytest.mark.validation
def test_kpoint_rks_hse06_d_shell_force_and_stress_match_fd(h_spd):
    _, fmiss = _force_vs_fd("rks_k", H2, h_spd, "HSE06", 1)
    _, smiss = _stress_vs_fd("rks_k", H2, h_spd, "HSE06", 1, ((0, 0), (2, 2), (0, 2)))
    print(f"k RKS HSE06 d-shell: force vs FD {fmiss:.2e}, stress vs FD {smiss:.2e}")
    assert fmiss < FD_BAR and smiss < STRESS_BAR


@pytest.mark.validation
def test_kpoint_uks_scan_d_shell_force_matches_fd(h_spd):
    _, miss = _force_vs_fd("uks_k", H3, h_spd, "SCAN", 2)
    print(f"k UKS SCAN d-shell force vs FD: {miss:.2e}")
    assert miss < FD_BAR


@pytest.mark.validation
def test_gamma_rks_cam_b3lyp_d_shell_force_and_stress_match_fd(h_spd):
    _, fmiss = _force_vs_fd("rks_g", H2, h_spd, CAM, 1)
    _, smiss = _stress_vs_fd("rks_g", H2, h_spd, CAM, 1, ((0, 0), (0, 2)))
    print(
        f"Gamma RKS CAM-B3LYP d-shell: force vs FD {fmiss:.2e}, stress vs FD {smiss:.2e}"
    )
    assert fmiss < FD_BAR and smiss < STRESS_BAR


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.mark.validation
def test_kpoint_rks_hse06_lih_force_and_stress_match_fd(sto3g):
    _, fmiss = _force_vs_fd("rks_k", LIH, sto3g, "HSE06", 1)
    _, smiss = _stress_vs_fd("rks_k", LIH, sto3g, "HSE06", 1, ((0, 0), (0, 2)))
    print(f"k RKS HSE06 LiH: force vs FD {fmiss:.2e}, stress vs FD {smiss:.2e}")
    assert fmiss < FD_BAR and smiss < STRESS_BAR


@pytest.mark.validation
def test_kpoint_uks_scan_lih2_force_matches_fd(sto3g):
    _, miss = _force_vs_fd("uks_k", LIH2, sto3g, "SCAN", 2)
    print(f"k UKS SCAN LiH2 force vs FD: {miss:.2e}")
    assert miss < FD_BAR
