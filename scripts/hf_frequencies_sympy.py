#!/usr/bin/env python3
"""Harmonic frequencies from a Cartesian Hessian as ferric computes them.

Derivation: crates/ferric-scf/theory.md, "Harmonic frequencies".
Code: crates/ferric-scf/src/frequencies.rs (frequencies_from_cartesian_hessian,
translation_rotation_basis, center_of_mass, detect_linear and the constants
AMU_TO_ELECTRON_MASS, AU_FREQ_TO_CM), transcribed below.

Checks:
  C1  AU_FREQ_TO_CM = E_h / (h c) from CODATA 2018 (relative 1e-10).
  C2  AMU_TO_ELECTRON_MASS = m_u / m_e from CODATA 2018 (relative 1e-10).
  S1  (sympy) the mass-weighted Hessian of a diatomic spring along z has the
      eigenvalues 0 and k (1/m1 + 1/m2) = k / mu on that axis.
  D1  Diatomic spring (k = 0.5 E_h/a0^2, masses of H and F) through the
      transcribed pipeline: 3N - 5 = 1 vibration; its wavenumber equals the SI
      route (1 / 2 pi c) sqrt(k_SI / mu_SI) with k_SI = k E_h / a0^2,
      mu_SI = mu m_u (relative 1e-9); the 5 projected modes are zero.
  P1  Nonlinear 4-atom geometry: the T/R basis is orthonormal with rank 6,
      P = 1 - T T^T is symmetric and idempotent, and P annihilates the raw
      mass-weighted translations and rotations.
  P2  Linear triatomic: rank 5, and detect_linear agrees.
  P3  Pair-spring potential at its minimum: the projected spectrum has six
      zeros and the vibrational eigenvalues equal the nonzero eigenvalues of
      the unprojected mass-weighted Hessian (projection leaves them intact).
  P4  Same springs away from the minimum (nonzero forces): the unprojected
      Hessian has nonzero rotational eigenvalues, the projected one has six
      exact zeros, identified as T/R by overlap (is_tr).

Mutations (each turns the script red; exact one-occurrence substitutions):
  AU_FREQ_TO_CM divided by 2 pi (angular vs ordinary frequency)   (C1, D1 fail)
  mass weighting 1/sqrt(m) -> 1/m                                 (D1 fails)
  rotation about x: (0, -r_z, r_y) -> (0, r_z, r_y)               (P1, P3 fail)
  projection sweep: the + t t^T (t^T H t) term flipped in sign     (P4 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# CODATA 2018 (independent of the code's constants).
E_H = 4.3597447222071e-18  # J
H_PLANCK = 6.62607015e-34  # J s (exact)
C_LIGHT = 299792458.0  # m/s (exact)
M_U = 1.66053906660e-27  # kg
M_E = 9.1093837015e-31  # kg
A0 = 5.29177210903e-11  # m

# Transcribed from frequencies.rs.
AMU_TO_ELECTRON_MASS = 1.0 / 5.485_799_090_65e-4
AU_FREQ_TO_CM = 219_474.631_363_2
LINEARITY_THRESH = 1.0e-6


def center_of_mass(coords, mass_au):
    total = sum(mass_au)
    com = np.zeros(3)
    for r, m in zip(coords, mass_au):
        com += m * r
    return com / total


def detect_linear(coords, mass_au):
    if len(coords) < 3:
        return True
    com = center_of_mass(coords, mass_au)
    inertia = np.zeros((3, 3))
    for r0, m in zip(coords, mass_au):
        r = r0 - com
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    ev = np.linalg.eigvalsh(inertia)
    mx = np.max(np.abs(ev))
    return mx > 0 and abs(ev[0]) / mx < LINEARITY_THRESH


def rotation_vectors(coords, mass_au):
    com = center_of_mass(coords, mass_au)
    out = []
    for k in range(3):
        v = np.zeros(3 * len(coords))
        for a, (r0, m) in enumerate(zip(coords, mass_au)):
            r = r0 - com
            cross = {
                0: [0.0, -r[2], r[1]],
                1: [r[2], 0.0, -r[0]],
                2: [-r[1], r[0], 0.0],
            }[k]
            v[3 * a : 3 * a + 3] = np.sqrt(m) * np.array(cross)
        out.append(v)
    return out


def translation_vectors(natoms, mass_au):
    out = []
    for k in range(3):
        v = np.zeros(3 * natoms)
        for a in range(natoms):
            v[3 * a + k] = np.sqrt(mass_au[a])
        out.append(v)
    return out


def translation_rotation_basis(coords, mass_au):
    raw = translation_vectors(len(coords), mass_au) + rotation_vectors(coords, mass_au)
    com = center_of_mass(coords, mass_au)
    scale = np.sqrt(sum(m * np.dot(r - com, r - com) for r, m in zip(coords, mass_au)))
    basis = []
    for idx, v in enumerate(raw):
        w = v.copy()
        norm0 = np.sqrt(w @ w)
        if norm0 <= 0.0:
            continue
        if idx >= 3 and norm0 * norm0 < LINEARITY_THRESH * scale * scale:
            continue
        for b in basis:
            w = w - b * (b @ w)
        norm = np.sqrt(w @ w)
        if norm / norm0 > 1.0e-6:
            basis.append(w / norm)
    return basis


def frequencies(coords, hess, masses_amu):
    """Transcription of frequencies_from_cartesian_hessian (steps 3-5)."""
    n = 3 * len(coords)
    mass_au = [m * AMU_TO_ELECTRON_MASS for m in masses_amu]
    inv_sqrt_m = np.array([1.0 / np.sqrt(mass_au[c // 3]) for c in range(n)])
    mw = hess * np.outer(inv_sqrt_m, inv_sqrt_m)
    basis = translation_rotation_basis(coords, mass_au)
    proj = mw.copy()
    for t in basis:
        ht = proj @ t
        tht = t @ ht
        proj = proj - (np.outer(t, ht) + np.outer(ht, t) - np.outer(t, t) * tht)
    proj = 0.5 * (proj + proj.T)
    evals, evecs = np.linalg.eigh(proj)
    is_tr = [sum((t @ evecs[:, k]) ** 2 for t in basis) > 0.5 for k in range(n)]

    def to_cm(lam):
        w = np.sqrt(abs(lam)) * AU_FREQ_TO_CM
        return -w if lam < 0 else w

    vib = [to_cm(evals[k]) for k in range(n) if not is_tr[k]]
    tr = [to_cm(evals[k]) for k in range(n) if is_tr[k]]
    return {
        "vib": vib,
        "tr": tr,
        "basis": basis,
        "mw": mw,
        "proj": proj,
        "mass_au": mass_au,
    }


def spring_hessian(coords, pairs):
    """Analytic Hessian of V = sum k/2 (r_ab - r0_ab)^2."""
    n = 3 * len(coords)
    hess = np.zeros((n, n))
    for a, b, k, r0 in pairs:
        d = coords[a] - coords[b]
        r = np.linalg.norm(d)
        u = d / r
        blk = k * (np.outer(u, u) + (1.0 - r0 / r) * (np.eye(3) - np.outer(u, u)))
        for i, j, sgn in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            hess[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += sgn * blk
    return hess


def main():
    # C1, C2: constants.
    cm_ref = E_H / (H_PLANCK * C_LIGHT * 100.0)
    rel = abs(AU_FREQ_TO_CM / cm_ref - 1.0)
    check(
        "C1 AU_FREQ_TO_CM = E_h/(h c) (claim C1)",
        rel < 1e-10,
        f"rel {rel:.1e} ref {cm_ref:.7f}",
    )
    rel = abs(AMU_TO_ELECTRON_MASS / (M_U / M_E) - 1.0)
    check("C2 AMU_TO_ELECTRON_MASS = m_u/m_e (claim C2)", rel < 1e-10, f"rel {rel:.1e}")

    # S1: symbolic diatomic.
    k, m1, m2 = sp.symbols("k m1 m2", positive=True)
    hmw = sp.Matrix([[k / m1, -k / sp.sqrt(m1 * m2)], [-k / sp.sqrt(m1 * m2), k / m2]])
    ev = list(hmw.eigenvals().keys())
    ok = any(sp.simplify(e) == 0 for e in ev) and any(
        sp.simplify(e - k * (1 / m1 + 1 / m2)) == 0 for e in ev
    )
    check("S1 diatomic mass-weighted eigenvalues {0, k/mu} (claim S1)", ok)

    # D1: diatomic through the pipeline vs the SI route.
    masses = [1.00794, 18.9984032]
    kau = 0.5
    coords = [np.array([0.1, -0.2, 0.3]), np.array([0.1, -0.2, 0.3 + 1.73])]
    res = frequencies(coords, spring_hessian(coords, [(0, 1, kau, 1.73)]), masses)
    mu_si = masses[0] * masses[1] / (masses[0] + masses[1]) * M_U
    k_si = kau * E_H / A0**2
    nu_ref = np.sqrt(k_si / mu_si) / (2.0 * np.pi * C_LIGHT * 100.0)
    ok = len(res["vib"]) == 1 and abs(res["vib"][0] / nu_ref - 1.0) < 1e-9
    check(
        "D1 diatomic wavenumber = (1/2 pi c) sqrt(k/mu) (SI route) (claim D1)",
        ok,
        f"ferric {res['vib']} ref {nu_ref:.6f}",
    )
    check(
        "D1 diatomic: 5 projected modes are zero (claim D1)",
        max(abs(x) for x in res["tr"]) < 1e-3,
    )

    # P1: nonlinear 4 atoms.
    rng = np.random.default_rng(3)
    coords = [rng.standard_normal(3) * 1.5 for _ in range(4)]
    masses = [12.011, 1.008, 15.999, 14.007]
    mass_au = [m * AMU_TO_ELECTRON_MASS for m in masses]
    basis = translation_rotation_basis(coords, mass_au)
    t = np.array(basis).T
    p = np.eye(12) - t @ t.T
    raw = translation_vectors(4, mass_au) + rotation_vectors(coords, mass_au)
    ok = (
        len(basis) == 6
        and np.max(np.abs(t.T @ t - np.eye(6))) < 1e-12
        and np.max(np.abs(p @ p - p)) < 1e-12
        and max(np.linalg.norm(p @ v) / np.linalg.norm(v) for v in raw) < 1e-12
    )
    check("P1 nonlinear: rank 6, orthonormal, P idempotent, P kills T/R (claim P1)", ok)
    # The raw rotations are rotations: they annihilate the gradient of any
    # pair-distance function (here each distance), i.e. d r_ab . v = 0.
    worst = 0.0
    inv_sqrt = np.repeat(1.0 / np.sqrt(mass_au), 3)
    for v in raw[3:]:
        dx = (v * inv_sqrt).reshape(4, 3)
        for a in range(4):
            for b in range(a):
                d = coords[a] - coords[b]
                worst = max(worst, abs(d @ (dx[a] - dx[b])))
    check(
        "P1 rotation vectors preserve every interatomic distance (claim P1)",
        worst < 1e-10,
        f"{worst:.1e}",
    )

    # P2: linear triatomic.
    lin = [np.array([0.0, 0.0, z]) for z in (-2.2, 0.0, 2.2)]
    lmass = [m * AMU_TO_ELECTRON_MASS for m in (15.999, 12.011, 15.999)]
    ok = len(translation_rotation_basis(lin, lmass)) == 5 and detect_linear(lin, lmass)
    check("P2 linear triatomic: rank 5 and detect_linear (claim P2)", ok)

    # P3: springs at their minimum.
    pairs = [
        (a, b, 0.3 + 0.1 * (a + b), np.linalg.norm(coords[a] - coords[b]))
        for a in range(4)
        for b in range(a)
    ]
    res = frequencies(coords, spring_hessian(coords, pairs), masses)
    ev_un = np.linalg.eigvalsh(res["mw"])
    nonzero = np.sort(ev_un)[6:]
    vib_ev = np.sort([np.sign(v) * (v / AU_FREQ_TO_CM) ** 2 for v in res["vib"]])
    rel = np.max(np.abs(vib_ev - nonzero) / np.abs(nonzero))
    ok = len(res["vib"]) == 6 and rel < 1e-9
    check(
        "P3 at a minimum the projection leaves the vibrations unchanged (claim P3)",
        ok,
        f"rel {rel:.1e}",
    )

    # P4: springs away from their minimum.
    pairs_off = [(a, b, kk, 0.8 * r0) for a, b, kk, r0 in pairs]
    res = frequencies(coords, spring_hessian(coords, pairs_off), masses)
    ev_un = np.linalg.eigvalsh(res["mw"])
    rot_un = max(
        abs(r @ res["mw"] @ r) / (r @ r)
        for r in rotation_vectors(coords, res["mass_au"])
    )
    ev_p = np.linalg.eigvalsh(res["proj"])
    zeros = np.sum(np.abs(ev_p) < 1e-12 * np.max(np.abs(ev_p)))
    ok = rot_un > 1e-8 and zeros == 6 and len(res["tr"]) == 6
    check(
        "P4 off-minimum: rotations carry curvature, projection zeroes 6 modes (claim P4)",
        ok,
        f"max rot curvature {rot_un:.1e}, zeros {zeros}",
    )

    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
