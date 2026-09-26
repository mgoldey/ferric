"""Independent numpy/PySCF reference for the Thole polarizable-embedding row.

Row (site/src/reference/validation.md, Anchors): "Thole polarizable embedding".
Consumer: crates/ferric-scf/tests/validation_thole.rs
Output:   testdata/reference/validation/thole/<case>_<basis>.json

No external polarizable-embedding code is installed (CPPE is not), so the
reference is a from-scratch numpy implementation of the model driven through
PySCF 2.13: PySCF supplies the AO integrals, the point-charge embedding
(`qmmm.mm_charge`) and the SCF driver; everything polarizable is below.

MODEL (read from crates/ferric-scf/src/polarizable.rs; line numbers at the
time of writing)
  * sites: every MM atom with alpha > 0, in atom order; alpha in Bohr^3
    (Python API: polarizabilities_angstrom3 * ANGSTROM_TO_BOHR**3,
    crates/ferric-python/src/lib.rs:832-834).
  * induction (polarizable.rs:359-413): dense solve B mu = E0 with
    B_ii = I/alpha_i, B_ij = -T_ij (i != j, pair not excluded).
  * E0_i = E_i^QM(D) + E_i^perm (polarizable.rs:545-552). E^QM = QM nuclei as
    point charges + electrons from the CURRENT total density. E^perm = every
    MM charge EXCEPT the one on site i itself and those on sites excluded
    from i (polarizable.rs:304-311, colocation by position). Permanent
    fields are NOT damped.
  * Thole damping (polarizable.rs:262-271): exponential Thole,
    u = r/(alpha_i alpha_j)^(1/6), v = a u (Thole's exponential density),
    lambda3 = 1 - (1 + v + v^2/2) exp(-v), lambda5 = lambda3 - v^3/6 exp(-v),
    default a = 2.1304
    (polarizable.rs:144), thole_a = None -> lambda3 = lambda5 = 1.
  * E_pol = -1/2 sum_i mu_i . E0_i (polarizable.rs:556-562), added to the
    SCF total energy as a standalone term (rhf.rs:1498, uhf.rs:940).
  * Fock term V_pol = -sum_i mu_i . <mu|(r - R_i)/|r - R_i|^3|nu>, no 1/2,
    mu recomputed from the current density every SCF iteration and added to
    every spin Fock (polarizable.rs:436-440, driver.rs:376-392).
  * dipole_zeta = 1e4 (polarizable.rs:148): ferric's field integrals are a
    p-shell Gaussian of width 0.01 Bohr; the reference uses exact
    point-dipole integrals.

THE TENSOR SIGN. Both ferric and this reference use the dipole field tensor
T_ij = (3 lambda5 r^ r^ - lambda3 I)/r^3 in mu = alpha (E0 + sum_j T_ij mu_j):
the field of a point dipole mu_j at R_i is -grad phi, phi(r) = mu_j.(r -
R_j)/|r - R_j|^3. The generator derives the tensor from -grad phi by finite
difference (`self_check_dipole_tensor`, refused if the closed form
disagrees). The control `negated_tensor` (`sign=-1`) is the opposite sign, a
diagnostic: a ferric energy that lands on it instead of on the reference
means the tensor sign has regressed.

TOTAL ENERGY (RHF; UHF identical with D = D_a + D_b):
  E = Tr(h_MM D) + 1/2 Tr(G[D] D) + E_nuc + E_nuc-MM + E_pol(D)
  where h_MM and E_nuc-MM come from qmmm.mm_charge. It is variational in D
  with Fock h_MM + G[D] + V_pol (V_pol = dE_pol/dD at stationary mu), so the
  patched `get_veff` returns G + V_pol and the patched `energy_elec` removes
  1/2 Tr(V_pol D) and adds E_pol. No MM-MM charge-charge term (ferric has none).

GRADIENTS. Reference gradients are 5-point central finite differences of the
fully reconverged total energy at h and 2h (must agree to FD_AGREE). An MM
atom's row moves its charge AND its polarizable site together (a QmmmAtom
carries both). Two analytic cross-checks at fixed (D, mu), both by FD of
closed-form functionals (no SCF):
  * `w_rows`: plain mm_charge MM gradient on the polarized density + d/dR of
    W = -mu.E0(R; D) + 1/2 mu.B(R).mu (stationary in mu, so the envelope
    theorem holds). REFUSED if it disagrees with the full FD by > W_AGREE.
  * `naive_rows`: the same with W replaced by the naive fixed-mu
    -1/2 mu.E0(R; D) (no factor 1, no T derivative). Recorded; the Rust test
    asserts it MISSES the FD (the negative control CLAUDE.md names).

CASES
  h2o_w4_excl   QM water + 4 TIP3P MM waters (O -0.834, H +0.417; alpha
                O 0.837, H 0.496 A^3), intramolecular site pairs excluded
                (the polarizable-force-field convention), a = 2.1304. RHF.
  h2o_w4_noexcl same geometry, no exclusions (the run_qmmm Python default,
                lib.rs:1244): intramolecular sites polarize each other and
                the Thole damping is live. alpha x ALPHA_SCALE_NOEXCL (0.3):
                at full alpha the damped B is indefinite (see the constant).
                RHF.
  nh2_w3_uhf    NH2 radical (2B1) + 3 MM waters, intramolecular exclusions.
                UHF. (OH was tried first and rejected: its pi-hole rotation is
                so soft that DIIS stalls at |g| ~1e-7 and two starts end
                3.5e-11 Ha apart, too noisy for a finite-difference gradient.)
  far_site      anchor: QM water + the 4 waters as fixed charges + a +1 ion,
                ONE polarizable site (alpha 10 A^3, no charge) 30 A away.
                Classical limit E_pol -> -1/2 alpha |E0|^2 with E0 the field
                of the NON-polarized embedded QM density + nuclei + MM
                charges at the site.
CONTROLS per case (energy, E_pol, dipoles): plain point-charge embedding,
no_damping (thole_a None), no_mutual (T = 0), negated_tensor (ferric's sign).

Run (light):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_thole.py [case ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "thole"
ROW_NAME = "Thole polarizable embedding"
GEN = "scripts/validation/gen_thole.py"
BASIS = "cc-pvdz"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
CONV_TOL_GRAD_UHF = 1e-8  # the common.run_open_shell default
THOLE_A = 2.1304
H_FD = 2e-3  # Bohr
FD_AGREE = 5e-8  # Ha/Bohr, |g(h) - g(2h)|
W_AGREE = 2e-7  # Ha/Bohr, envelope-theorem rows vs full FD
H_FIXED = 1e-5  # Bohr, FD of closed-form functionals at fixed (D, mu)
A2B = common.ANGSTROM_TO_BOHR
ALPHA_A3 = {"O": 0.837, "H": 0.496}
Q_TIP3P = {"O": -0.834, "H": 0.417}
FAR_ALPHA_A3 = 10.0
FAR_DIST_A = 30.0
# No-exclusion case: intramolecular O-H site pairs sit 1.8 Bohr apart. At
# 0.5 x ALPHA_A3 the damped induction matrix B is comfortably positive
# definite (min eigenvalue 0.287) AND so is the undamped one (0.028), so the
# no-damping control is a physical state rather than a polarization
# catastrophe. (At full ALPHA_A3 the damped B is still positive definite,
# 0.123, but the undamped one is not, -0.200.)
ALPHA_SCALE_NOEXCL = 0.5


# ---------------------------------------------------------------------------
# Geometry (Angstrom in, Bohr out)
# ---------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def tip3p(o, bisector, normal_hint):
    """O, H, H (Angstrom) of a TIP3P-geometry water (r 0.9572 A, 104.52 deg)."""
    r, half = 0.9572, np.radians(104.52) / 2.0
    b = _unit(bisector)
    n = np.asarray(normal_hint, dtype=float)
    n = _unit(n - n.dot(b) * b)
    t = np.cross(n, b)
    o = np.asarray(o, dtype=float)
    return [
        o,
        o + r * (np.cos(half) * b + np.sin(half) * t),
        o + r * (np.cos(half) * b - np.sin(half) * t),
    ]


def qm_angstrom(name):
    syms, c = common.read_xyz(common.MOL_DIR / name)
    return syms, [np.array(x) / A2B for x in c]


def waters_around_h2o():
    _, (o, h1, h2) = qm_angstrom("h2o.xyz")
    u1, u2 = _unit(h1 - o), _unit(h2 - o)
    bis = _unit(h1 + h2 - 2.0 * o)
    w = [
        # accepts from H1, tilted out of the QM plane
        tip3p(h1 + 1.95 * u1, u1 + np.array([0.3, 0.0, 0.1]), [1.0, 0.3, 0.2]),
        # accepts from H2, different tilt (no mirror symmetry)
        tip3p(h2 + 2.05 * u2, u2 + np.array([-0.2, 0.1, 0.3]), [0.2, 1.0, -0.4]),
        # donates to the O lone pair: its O sits 2.9 A out along -bisector
        tip3p(
            o + 2.9 * _unit(-bis + np.array([0.5, 0.0, 0.0])),
            _unit(-bis + np.array([0.5, 0.0, 0.0])) + np.array([0.0, 0.4, 0.2]),
            [0.1, 0.2, 1.0],
        ),
        # a second-shell water
        tip3p(np.array([3.3, 1.6, 2.1]), [-0.6, 0.3, -0.5], [0.4, -0.2, 1.0]),
    ]
    return w


def waters_around_nh2():
    _, (n, h1, h2) = qm_angstrom("nh2.xyz")
    u1, u2 = _unit(h1 - n), _unit(h2 - n)
    bis = _unit(h1 + h2 - 2.0 * n)
    return [
        # accepts from H1
        tip3p(h1 + 2.1 * u1, u1 + np.array([0.3, 0.1, 0.0]), [1.0, 0.2, 0.3]),
        # donates to the N lone pair side, tilted out of plane
        tip3p(
            n + 3.0 * _unit(-bis + np.array([0.4, 0.5, 0.0])),
            _unit(-bis + np.array([0.4, 0.5, 0.0])) + np.array([0.2, 0.0, 0.3]),
            [0.0, 1.0, 0.3],
        ),
        # beside H2, second shell
        tip3p(
            h2 + 2.6 * _unit(u2 + np.array([0.0, 0.0, 0.6])),
            [-0.3, 0.9, 0.2],
            [1.0, 0.0, 0.4],
        ),
    ]


def mm_from_waters(waters, polarizable=True, alpha_scale=1.0):
    """[{symbol, q, xyz_bohr, alpha_angstrom3, alpha_bohr3}] and intramolecular
    site-index exclusion pairs (all MM atoms are sites when polarizable)."""
    atoms, excl = [], []
    for w in waters:
        base = len(atoms)
        for k, p in enumerate(w):
            s = "O" if k == 0 else "H"
            a3 = ALPHA_A3[s] * alpha_scale if polarizable else 0.0
            atoms.append(
                {
                    "symbol": s,
                    "q": Q_TIP3P[s],
                    "xyz_bohr": [float(c) * A2B for c in p],
                    "alpha_angstrom3": a3,
                    "alpha_bohr3": a3 * A2B**3,
                }
            )
        excl += [(base, base + 1), (base, base + 2), (base + 1, base + 2)]
    return atoms, excl


def check_min_distance(qm_bohr, mm, floor_bohr=1.5 * A2B):
    qm = np.asarray(qm_bohr)
    d = min(
        float(np.min(np.linalg.norm(qm - np.array(a["xyz_bohr"]), axis=1))) for a in mm
    )
    if d < floor_bohr:
        raise RuntimeError(
            f"an MM atom is {d:.3f} Bohr from a QM atom (< {floor_bohr:.3f})"
        )
    return d


# ---------------------------------------------------------------------------
# The polarizable model
# ---------------------------------------------------------------------------


def thole_tensor_phys(ri, rj, ai, aj, a, sign=1.0):
    """PHYSICAL Thole-damped dipole field tensor: field at ri of a dipole mu at
    rj is T mu, T = (3 l5 r^r^ - l3 I)/r^3. `sign=-1` gives ferric's tensor."""
    d = np.asarray(ri) - np.asarray(rj)
    r = float(np.linalg.norm(d))
    rh = d / r
    if a is None:
        l3 = l5 = 1.0
    else:
        u = r / (ai * aj) ** (1.0 / 6.0)
        v = a * u
        ex = np.exp(-v)
        l3 = 1.0 - (1.0 + v + 0.5 * v * v) * ex
        l5 = l3 - v**3 / 6.0 * ex
    return sign * (3.0 * l5 * np.outer(rh, rh) - l3 * np.eye(3)) / r**3


class Model:
    """Sites, permanent charges, exclusions and the induction solve."""

    def __init__(self, mm, exclusions, thole_a=THOLE_A, t_sign=1.0, mutual=True):
        self.mm = mm
        self.site_atom = [k for k, a in enumerate(mm) if a["alpha_bohr3"] > 0.0]
        self.pos = np.array([mm[k]["xyz_bohr"] for k in self.site_atom]).reshape(-1, 3)
        self.alpha = np.array([mm[k]["alpha_bohr3"] for k in self.site_atom])
        self.q = np.array([a["q"] for a in mm])
        self.qpos = np.array([a["xyz_bohr"] for a in mm])
        self.excl = {tuple(sorted(p)) for p in exclusions}
        self.thole_a, self.t_sign, self.mutual = thole_a, t_sign, mutual

    def excluded(self, i, j):
        return tuple(sorted((i, j))) in self.excl

    def e_perm(self):
        """Field of the MM charges at each site; site i skips the charge on
        its own atom and on the atoms of sites excluded from it."""
        out = np.zeros((len(self.site_atom), 3))
        site_of_atom = {k: s for s, k in enumerate(self.site_atom)}
        for i, ri in enumerate(self.pos):
            for c, (qc, rc) in enumerate(zip(self.q, self.qpos)):
                j = site_of_atom.get(c)
                if j is not None and (j == i or self.excluded(i, j)):
                    continue
                d = ri - rc
                out[i] += qc * d / np.linalg.norm(d) ** 3
        return out

    def bmat(self):
        n = len(self.site_atom)
        b = np.zeros((3 * n, 3 * n))
        for i in range(n):
            b[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = np.eye(3) / self.alpha[i]
        if not self.mutual:
            return b
        for i in range(n):
            for j in range(n):
                if i == j or self.excluded(i, j):
                    continue
                t = thole_tensor_phys(
                    self.pos[i],
                    self.pos[j],
                    self.alpha[i],
                    self.alpha[j],
                    self.thole_a,
                    self.t_sign,
                )
                b[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -t
        return b

    def induce(self, mol, dm_tot):
        """(mu (n,3), e_pol, v_pol (nao,nao), e0 (n,3))."""
        nao = mol.nao_nr()
        if len(self.site_atom) == 0:
            return np.zeros((0, 3)), 0.0, np.zeros((nao, nao)), np.zeros((0, 3))
        fints = field_integrals(mol, self.pos)
        e0 = qm_field(mol, dm_tot, self.pos, fints) + self.e_perm()
        mu = np.linalg.solve(self.bmat(), e0.reshape(-1)).reshape(-1, 3)
        e_pol = -0.5 * float(np.sum(mu * e0))
        v_pol = -np.einsum("ic,icmn->mn", mu, fints)
        return mu, e_pol, v_pol, e0


def field_integrals(mol, points):
    """F[i, c] = <mu|(r - R_i)_c/|r - R_i|^3|nu> = iprinv + iprinv^T."""
    out = np.empty((len(points), 3, mol.nao_nr(), mol.nao_nr()))
    for i, p in enumerate(points):
        with mol.with_rinv_origin(p):
            ip = mol.intor("int1e_iprinv", comp=3)
        out[i] = ip + ip.transpose(0, 2, 1)
    return out


def qm_field(mol, dm_tot, points, fints=None):
    """Electric field of the QM nuclei (point charges) + electrons at points."""
    if fints is None:
        fints = field_integrals(mol, points)
    e = np.einsum("icmn,nm->ic", fints, dm_tot)
    for z, ra in zip(mol.atom_charges(), mol.atom_coords()):
        d = np.asarray(points) - ra
        e += z * d / np.linalg.norm(d, axis=1)[:, None] ** 3
    return e


def qm_potential(mol, dm_tot, p):
    with mol.with_rinv_origin(p):
        v = mol.intor("int1e_rinv")
    out = -float(np.einsum("mn,nm->", v, dm_tot))
    for z, ra in zip(mol.atom_charges(), mol.atom_coords()):
        out += z / np.linalg.norm(np.asarray(p) - ra)
    return out


# ---------------------------------------------------------------------------
# Self-checks (refuse to write a reference if any fails)
# ---------------------------------------------------------------------------


def self_check_dipole_tensor():
    """Closed-form T_phys (undamped) vs -grad of the dipole potential."""
    rj = np.array([0.1, -0.3, 0.2])
    ri = np.array([1.3, 2.2, -3.1])
    mu = np.array([0.3, -0.7, 0.5])

    def phi(r):
        d = r - rj
        return mu @ d / np.linalg.norm(d) ** 3

    h = 1e-5
    e_fd = np.array([-(phi(ri + h * e) - phi(ri - h * e)) / (2 * h) for e in np.eye(3)])
    e_t = thole_tensor_phys(ri, rj, 1.0, 1.0, None) @ mu
    err = float(np.abs(e_fd - e_t).max())
    if err > 1e-8:
        raise RuntimeError(f"dipole tensor self-check failed: {err:.2e}")
    ferric_like = -e_t
    return {
        "check": "T_phys(undamped) . mu == -grad[mu.(r-Rj)/|r-Rj|^3] by central FD",
        "max_abs_err": err,
        "ferric_T_times_mu_over_physical_field": float(
            ferric_like @ e_fd / (e_fd @ e_fd)
        ),
    }


def self_check_field(mol, dm, point):
    """Electron+nuclear field from the integrals vs -grad of the ESP."""
    h = 1e-4
    e_fd = np.array(
        [
            -(
                qm_potential(mol, dm, point + h * e)
                - qm_potential(mol, dm, point - h * e)
            )
            / (2 * h)
            for e in np.eye(3)
        ]
    )
    e_int = qm_field(mol, dm, np.array([point]))[0]
    err = float(np.abs(e_fd - e_int).max())
    if err > 1e-8:
        raise RuntimeError(f"field-integral self-check failed: {err:.2e}")
    return {"check": "field integrals == -grad(ESP) by central FD", "max_abs_err": err}


# ---------------------------------------------------------------------------
# SCF with the polarizable term
# ---------------------------------------------------------------------------


def build_mol(symbols, coords_bohr, spin=0):
    from pyscf import gto

    bas, cart = common.pyscf_basis(BASIS, symbols)
    assert not cart
    return gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords_bohr),
        unit="Bohr",
        basis=bas,
        cart=False,
        spin=spin,
        verbose=0,
    )


def plain_mf(mol, mm, uhf):
    from pyscf import qmmm, scf

    base = scf.UHF(mol) if uhf else scf.RHF(mol)
    mf = qmmm.mm_charge(
        base,
        np.array([a["xyz_bohr"] for a in mm]),
        np.array([a["q"] for a in mm]),
        unit="Bohr",
    )
    mf.conv_tol, mf.verbose = CONV_TOL, 0
    mf.max_cycle = 300
    mf.conv_tol_grad = CONV_TOL_GRAD_UHF if uhf else CONV_TOL_GRAD
    mf.direct_scf = False  # get_veff must never add onto a previous vhf
    return mf


def pol_mf(mol, mm, model, uhf):
    """mm_charge SCF whose Fock carries V_pol and whose energy carries E_pol."""
    from pyscf import lib

    mf = plain_mf(mol, mm, uhf)
    cls = mf.__class__

    def dtot(dm):
        dm = np.asarray(dm)
        return dm if dm.ndim == 2 else dm[0] + dm[1]

    def get_veff(mol_=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        mol_ = mf.mol if mol_ is None else mol_
        dm = mf.make_rdm1() if dm is None else dm
        vhf = np.asarray(cls.get_veff(mf, mol_, dm))
        mu, e_pol, v_pol, _ = model.induce(mol_, dtot(dm))
        return lib.tag_array(vhf + v_pol, v_pol=v_pol, e_pol=e_pol)

    def energy_elec(dm=None, h1e=None, vhf=None):
        dm = mf.make_rdm1() if dm is None else dm
        if vhf is None or getattr(vhf, "v_pol", None) is None:
            vhf = get_veff(mf.mol, dm)
        plain = np.asarray(vhf) - vhf.v_pol
        e, e2 = cls.energy_elec(mf, dm, h1e, plain)
        return e + vhf.e_pol, e2 + vhf.e_pol

    mf.get_veff = get_veff
    mf.energy_elec = energy_elec
    mf._dtot = dtot
    return mf


def run(mol, mm, model, uhf, dm0=None):
    # Retry once from the default guess if the neighbouring density stalls;
    # a second failure is a hard error.
    for guess in (dm0, None) if dm0 is not None else (None,):
        mf = pol_mf(mol, mm, model, uhf)
        e = mf.kernel(dm0=guess)
        if mf.converged:
            break
    if not mf.converged:
        raise RuntimeError("polarizable SCF did not converge")
    dm = mf.make_rdm1()
    mu, e_pol, _, e0 = model.induce(mol, mf._dtot(dm))
    return float(e), float(e_pol), mu, e0, dm, mf


def run_plain(mol, mm, uhf, dm0=None):
    mf = plain_mf(mol, mm, uhf)
    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("plain embedded SCF did not converge")
    return float(e), mf.make_rdm1(), mf


def displaced_mm(mm, k, c, step):
    out = [dict(a) for a in mm]
    xyz = list(out[k]["xyz_bohr"])
    xyz[c] += step
    out[k]["xyz_bohr"] = xyz
    return out


STENCIL = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]


def fd_qm(symbols, coords, spin, mm, model_kw, uhf, h, dm0):
    c0 = np.array(coords)
    g = np.zeros_like(c0)
    for a in range(c0.shape[0]):
        for k in range(3):
            acc = 0.0
            for s, w in STENCIL:
                c = c0.copy()
                c[a, k] += s * h
                mol = build_mol(symbols, c.tolist(), spin)
                acc += w * run(mol, mm, Model(mm, **model_kw), uhf, dm0)[0]
            g[a, k] = acc / h
    return g


def fd_mm(mol, mm, model_kw, uhf, h, dm0):
    g = np.zeros((len(mm), 3))
    for a in range(len(mm)):
        for k in range(3):
            acc = 0.0
            for s, w in STENCIL:
                mm_d = displaced_mm(mm, a, k, s * h)
                acc += w * run(mol, mm_d, Model(mm_d, **model_kw), uhf, dm0)[0]
            g[a, k] = acc / h
    return g


def fixed_rows(mol, mm, model_kw, uhf, dm, mu):
    """Envelope-theorem MM rows at fixed (D, mu): (w_rows, naive_rows)."""
    mf = plain_mf(mol, mm, uhf)
    g = mf.nuc_grad_method()
    plain = np.asarray(g.grad_hcore_mm(dm) + g.grad_nuc_mm())
    dtot = dm if np.asarray(dm).ndim == 2 else dm[0] + dm[1]

    def parts(mm_d):
        m = Model(mm_d, **model_kw)
        e0 = qm_field(mol, dtot, m.pos) + m.e_perm()
        w = -float(np.sum(mu * e0)) + 0.5 * float(
            mu.reshape(-1) @ m.bmat() @ mu.reshape(-1)
        )
        naive = -0.5 * float(np.sum(mu * e0))
        return w, naive

    w_rows, naive_rows = plain.copy(), plain.copy()
    for a in range(len(mm)):
        for k in range(3):
            wp, np_ = parts(displaced_mm(mm, a, k, H_FIXED))
            wm, nm = parts(displaced_mm(mm, a, k, -H_FIXED))
            w_rows[a, k] += (wp - wm) / (2 * H_FIXED)
            naive_rows[a, k] += (np_ - nm) / (2 * H_FIXED)
    return w_rows, naive_rows, plain


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def payload_head(
    case, xyzname, symbols, coords, mol, mm, excl, thole_a, method, spin, extra
):
    import pyscf

    xyz = common.MOL_DIR / xyzname
    basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)
    return {
        "row": ROW_NAME,
        "system": case,
        "basis": BASIS,
        "method": method,
        "charge": 0,
        "multiplicity": spin + 1,
        "nao": int(mol.nao_nr()),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "atoms": [{"symbol": s, "xyz_bohr": c} for s, c in zip(symbols, coords)],
        "mm_atoms": mm,
        "exclusions": [list(p) for p in excl],
        "thole_a": thole_a,
        "tensor_convention": "reference: physical T = (3 l5 r^r^ - l3 I)/r^3, mu = alpha (E0 + T mu)",
        "units": "Bohr / Hartree / a.u.; gradients are dE/dR; MM rows move charge AND site together",
        "provenance": common.provenance(
            code="numpy Thole model on PySCF integrals + qmmm.mm_charge",
            version=pyscf.__version__,
            keywords=(
                f"{method.upper()} exact 4-index J/K, conv_tol {CONV_TOL}, conv_tol_grad "
                f"{CONV_TOL_GRAD}; qmmm.mm_charge(unit='Bohr'); polarization in patched "
                "get_veff/energy_elec (see generator docstring); field integrals "
                "int1e_iprinv + transpose (exact point dipole)"
            ),
            basis_name=BASIS,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator=GEN,
            extra={"basis_check": basis_check, **extra},
        ),
    }


def control(mol, mm, uhf, dm0, ref_e, ref_mu, **model_kw):
    try:
        e, e_pol, mu, _, _, _ = run(mol, mm, Model(mm, **model_kw), uhf, dm0)
    except (RuntimeError, np.linalg.LinAlgError) as err:
        return {"converged": False, "error": str(err)}
    return {
        "converged": True,
        "energy": e,
        "e_pol": e_pol,
        "induced_dipoles": mu.tolist(),
        "abs_diff_energy_vs_reference": abs(e - ref_e),
        "max_abs_diff_dipole_vs_reference": float(np.abs(mu - ref_mu).max()),
    }


def full_case(case, xyzname, mm, excl, uhf, spin, mm_fd=True):
    symbols, coords = common.read_xyz(common.MOL_DIR / xyzname)
    check_min_distance(coords, mm)
    mol = build_mol(symbols, coords, spin)
    model_kw = {"exclusions": excl, "thole_a": THOLE_A}
    b_min = float(np.linalg.eigvalsh(Model(mm, **model_kw).bmat()).min())
    if b_min <= 0.0:
        raise RuntimeError(
            f"{case}: induction matrix B is indefinite (min eig {b_min:.3e})"
        )
    e_plain, dm_plain, mf_plain = run_plain(mol, mm, uhf)
    e, e_pol, mu, e0, dm, _ = run(mol, mm, Model(mm, **model_kw), uhf, dm_plain)
    dtot = dm if np.asarray(dm).ndim == 2 else dm[0] + dm[1]
    stability = None
    if uhf:
        # PySCF's stability() Hessian has no polarization response, so the check
        # is on the plain point-charge-embedded UHF state the polarized SCF is
        # started from; refused if that state is unstable.
        mo_i, stable = common._stability_status(mf_plain)
        s2 = float(mf_plain.spin_square()[0])
        stability = {
            "internal_stable": bool(stable),
            "checked_state": "plain point-charge-embedded UHF (no polarization kernel)",
            "s2_plain": s2,
        }
        if not stable:
            raise RuntimeError(
                f"{case}: plain embedded UHF state is internally unstable"
            )
    checks = {
        "dipole_tensor": self_check_dipole_tensor(),
        "field_integrals": self_check_field(mol, dtot, np.array(mm[0]["xyz_bohr"])),
    }
    head = payload_head(
        case,
        xyzname,
        symbols,
        coords,
        mol,
        mm,
        excl,
        THOLE_A,
        "uhf" if uhf else "rhf",
        spin,
        {"self_checks": checks},
    )
    out = {
        **head,
        "energy": e,
        "e_pol": e_pol,
        "energy_plain_embedding": e_plain,
        "induced_dipoles": mu.tolist(),
        "e0_at_sites": e0.tolist(),
        "induction_matrix_min_eigenvalue": b_min,
        "uhf_stability": stability,
    }
    g_h = fd_qm(symbols, coords, spin, mm, model_kw, uhf, H_FD, dm)
    g_2h = fd_qm(symbols, coords, spin, mm, model_kw, uhf, 2 * H_FD, dm)
    d_qm = float(np.abs(g_h - g_2h).max())
    if d_qm > FD_AGREE:
        raise RuntimeError(f"{case}: QM FD h vs 2h disagree by {d_qm:.2e}")
    out["qm_gradient_fd"] = g_h.tolist()
    out["qm_gradient_fd_2h"] = g_2h.tolist()
    fd_meta = {
        "stencil": "5-point central",
        "step_bohr": H_FD,
        "qm_max_abs_diff_h_vs_2h": d_qm,
    }
    if mm_fd:
        m_h = fd_mm(mol, mm, model_kw, uhf, H_FD, dm)
        m_2h = fd_mm(mol, mm, model_kw, uhf, 2 * H_FD, dm)
        d_mm = float(np.abs(m_h - m_2h).max())
        if d_mm > FD_AGREE:
            raise RuntimeError(f"{case}: MM FD h vs 2h disagree by {d_mm:.2e}")
        w_rows, naive_rows, plain_rows = fixed_rows(mol, mm, model_kw, uhf, dm, mu)
        d_w = float(np.abs(w_rows - m_h).max())
        if d_w > W_AGREE:
            raise RuntimeError(
                f"{case}: envelope-theorem MM rows miss the FD by {d_w:.2e}"
            )
        d_naive = float(np.abs(naive_rows - m_h).max())
        out["mm_gradient_fd"] = m_h.tolist()
        out["mm_gradient_fd_2h"] = m_2h.tolist()
        out["mm_gradient_w_fixed_mu"] = w_rows.tolist()
        out["mm_gradient_naive_fixed_mu"] = naive_rows.tolist()
        out["mm_gradient_charge_only_on_polarized_density"] = plain_rows.tolist()
        fd_meta.update(
            {
                "mm_max_abs_diff_h_vs_2h": d_mm,
                "mm_w_rows_max_abs_diff_vs_fd": d_w,
                "mm_naive_rows_max_abs_diff_vs_fd": d_naive,
            }
        )
    out["fd"] = fd_meta
    ctrl = {
        "plain_embedding": {
            "energy": e_plain,
            "abs_diff_energy_vs_reference": abs(e_plain - e),
        },
        "no_damping": control(mol, mm, uhf, dm, e, mu, exclusions=excl, thole_a=None),
        "no_mutual": control(
            mol, mm, uhf, dm, e, mu, exclusions=excl, thole_a=THOLE_A, mutual=False
        ),
        "negated_tensor": control(
            mol, mm, uhf, dm, e, mu, exclusions=excl, thole_a=THOLE_A, t_sign=-1.0
        ),
    }
    out["controls"] = ctrl
    path = common.write_reference(ROW, case, BASIS, out)
    summary = " ".join(
        f"{k}:{v.get('abs_diff_energy_vs_reference', float('nan')):.2e}"
        for k, v in ctrl.items()
    )
    print(
        f"{path.name}: E {e:.10f} E_pol {e_pol:+.8e} |E-E_plain| {abs(e - e_plain):.2e} "
        f"max|mu| {np.abs(mu).max():.3e} fd(qm) {d_qm:.1e} "
        + (
            f"fd(mm) {fd_meta['mm_max_abs_diff_h_vs_2h']:.1e} W-vs-FD "
            f"{fd_meta['mm_w_rows_max_abs_diff_vs_fd']:.1e} naive-vs-FD "
            f"{fd_meta['mm_naive_rows_max_abs_diff_vs_fd']:.1e} "
            if mm_fd
            else ""
        )
        + f"controls |dE| {summary}"
    )
    return path


def case_h2o_w4_excl():
    mm, excl = mm_from_waters(waters_around_h2o())
    return full_case("h2o_w4_excl", "h2o.xyz", mm, excl, uhf=False, spin=0)


def case_h2o_w4_noexcl():
    mm, _ = mm_from_waters(waters_around_h2o(), alpha_scale=ALPHA_SCALE_NOEXCL)
    return full_case("h2o_w4_noexcl", "h2o.xyz", mm, [], uhf=False, spin=0)


def case_nh2_w3_uhf():
    mm, excl = mm_from_waters(waters_around_nh2())
    return full_case("nh2_w3_uhf", "nh2.xyz", mm, excl, uhf=True, spin=1, mm_fd=False)


def case_far_site():
    symbols, coords = common.read_xyz(common.MOL_DIR / "h2o.xyz")
    mm, _ = mm_from_waters(waters_around_h2o(), polarizable=False)
    ion = np.array([-3.0, 1.2, 2.6]) * A2B
    mm.append(
        {
            "symbol": "Na",
            "q": 1.0,
            "xyz_bohr": ion.tolist(),
            "alpha_angstrom3": 0.0,
            "alpha_bohr3": 0.0,
        }
    )
    direction = _unit([0.6, -0.5, 0.62])
    site = (np.mean(np.array(coords), axis=0) + FAR_DIST_A * A2B * direction).tolist()
    mm.append(
        {
            "symbol": "X",
            "q": 0.0,
            "xyz_bohr": site,
            "alpha_angstrom3": FAR_ALPHA_A3,
            "alpha_bohr3": FAR_ALPHA_A3 * A2B**3,
        }
    )
    check_min_distance(coords, mm)
    mol = build_mol(symbols, coords)
    e_plain, dm_plain, _ = run_plain(mol, mm, False)
    model = Model(mm, exclusions=[], thole_a=THOLE_A)
    e, e_pol, mu, e0_scf, dm, _ = run(mol, mm, model, False, dm_plain)
    # Classical limit: field of the NON-polarized embedded system at the site.
    e0 = qm_field(mol, dm_plain, model.pos) + model.e_perm()
    alpha = model.alpha[0]
    e_cl = -0.5 * alpha * float(e0[0] @ e0[0])
    rel = abs(e_pol - e_cl) / abs(e_cl)
    if rel > 1e-3:
        raise RuntimeError(
            f"far_site: SCF E_pol {e_pol:.6e} vs classical {e_cl:.6e} (rel {rel:.1e})"
        )
    head = payload_head(
        "far_site",
        "h2o.xyz",
        symbols,
        coords,
        mol,
        mm,
        [],
        THOLE_A,
        "rhf",
        0,
        {"self_checks": {"dipole_tensor": self_check_dipole_tensor()}},
    )
    out = {
        **head,
        "energy": e,
        "e_pol": e_pol,
        "energy_plain_embedding": e_plain,
        "energy_difference_pol_minus_plain": e - e_plain,
        "induced_dipoles": mu.tolist(),
        "far_site": {
            "distance_angstrom_from_qm_centroid": FAR_DIST_A,
            "alpha_bohr3": float(alpha),
            "e0_nonpolarized_at_site": e0[0].tolist(),
            "e_pol_classical_limit": e_cl,
            "rel_diff_scf_e_pol_vs_classical": rel,
            "mu_classical": (alpha * e0[0]).tolist(),
        },
    }
    path = common.write_reference(ROW, "far_site", BASIS, out)
    print(
        f"{path.name}: E_pol {e_pol:+.10e} classical {e_cl:+.10e} rel {rel:.2e} "
        f"|E0| {np.linalg.norm(e0[0]):.3e} E-E_plain {e - e_plain:+.3e}"
    )
    return path


CASES = {
    "h2o_w4_excl": case_h2o_w4_excl,
    "h2o_w4_noexcl": case_h2o_w4_noexcl,
    "nh2_w3_uhf": case_nh2_w3_uhf,
    "far_site": case_far_site,
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    only = [a for a in argv if not a.startswith("-")]
    for c in only:
        if c not in CASES:
            raise SystemExit(f"unknown system {c}; choose from {sorted(CASES)}")
    written = []
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        written.append(fn())
    print(f"GEN_THOLE_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
