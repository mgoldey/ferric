#!/usr/bin/env python3
"""Order-by-order SR/LR coupling content of SR-MP2 + LR-RPA formulations.

One-mode ring model: a single particle-hole channel with gap D and kernel
k = k_s + k_l (short-range + long-range). dRPA has the closed form
E_c = (sqrt(D(D+2k)) - D - k)/2; MP2 is a generic quadratic form -c*k^2.
Reproduces the derivation behind docs/papers/sr-mp2-lr-rpa-methodology.md.
"""
import sympy as sp

D, ks, kl, c = sp.symbols('Delta k_s k_l c', positive=True)
k = sp.Symbol('k', positive=True)
N = 6

Erpa = (sp.sqrt(D*(D + 2*k)) - D - k)/2
ser = sp.series(Erpa, k, 0, N).removeO().expand()
ring_n = {n: ser.coeff(k, n) for n in range(2, N)}
rings = lambda kr: sum(ring_n[n]*kr**n for n in range(2, N))
E2 = lambda kr: -c*kr**2

A_naive = E2(ks) + rings(kl)
B_delta = E2(ks+kl) + rings(kl) - ring_n[2]*kl**2
T_coup  = E2(ks+kl) + (rings(ks+kl) - ring_n[2]*(ks+kl)**2) - (rings(ks) - ring_n[2]*ks**2)

# target: exact 2nd order + all rings >= 3rd order EXCEPT pure-SR rings
target = sp.expand(E2(ks+kl) + rings(ks+kl) - ring_n[2]*(ks+kl)**2
                   - sum(ring_n[n]*ks**n for n in range(3, N)))

print("dRPA ring series:", ser)
for name, f in [("naive A", A_naive), ("Delta-form B", B_delta), ("coupled T", T_coup)]:
    miss = sp.expand(target - sp.expand(f))
    print(f"target - {name}: {sp.factor(miss) if miss else 0}")
