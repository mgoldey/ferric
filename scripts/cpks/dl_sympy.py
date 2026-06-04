#!/usr/bin/env python3
"""Traceable SymPy derivation of the MP2 relaxed-density polarizability response.

Strategy: build the FULL relaxed-MP2 α on a SMALL explicit closed-shell system
with symbolic MO integrals, orbital energies, and a dipole-field perturbation F.
Everything (t2, P_oo, P_vv, Lagrangian, Z-vector, relaxed density, α) is a SymPy
expression. We then:
  (1) compute α(F) and α = -d^2 E/dF^2 = -d/dF Tr[D_relax(F) r]  symbolically,
  (2) compare to the term-by-term "response" assembly the Rust uses, isolating
      which ∂L term is wrong.

This is the tracable reference: a tiny system (nocc=1, nvir=2) where the full
relaxed α is computable in closed form, validating the analytic ∂L structure
WITHOUT finite difference.
"""
import sympy as sp

# --- tiny closed-shell system: 1 occ, 2 vir (3 spatial MOs) ---
nocc, nvir = 1, 2
nmo = nocc + nvir
O = list(range(nocc))            # occ indices 0
V = list(range(nocc, nmo))       # vir indices 1,2

# Orbital energies (symbolic, field-independent baseline).
eps = sp.symbols(f'e0:{nmo}', real=True)

# Field strength.
F = sp.symbols('F', real=True)

# Dipole matrix in MO basis: r_pq (symmetric). Field couples h += -F r.
r = sp.Matrix(nmo, nmo, lambda p, q: sp.Symbol(f'r{min(p,q)}{max(p,q)}', real=True))

# MO 2e integrals (pq|rs) as a symbolic 4-index dict with 8-fold symmetry.
def eri_sym(p, q, rr, s):
    # canonical ordering for 8-fold symmetry (chemist notation (pq|rs))
    a, b = (p, q) if p <= q else (q, p)
    c, d = (rr, s) if rr <= s else (s, rr)
    (a, b), (c, d) = sorted([(a, b), (c, d)])
    return sp.Symbol(f'I_{a}{b}_{c}{d}', real=True)

# Perturbed orbitals at first order: canonical CPHF would give U; for the tiny
# closed-form check we treat the field as making h_pq = eps_p δ_pq - F r_pq and
# solve the 3x3 (well, response) directly. We build the FOCK in MO at field F:
Fock = sp.Matrix(nmo, nmo, lambda p, q: (eps[p] if p == q else 0) - F * r[p, q])

print("Framework scaffolded. nocc=%d nvir=%d nmo=%d" % (nocc, nvir, nmo))
print("Fock(F) =")
sp.pprint(Fock)
